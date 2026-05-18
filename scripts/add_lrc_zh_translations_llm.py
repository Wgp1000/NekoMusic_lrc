#!/usr/bin/env python3
"""
用大模型为 LRC 添加 {"中文对照"} 行（OpenAI 兼容 API）。

优化要点：
- 本地预过滤：中文原文、拟声/口号、元数据行不调用 LLM
- 对照仅输出简体中文，拒绝中译中重复
- 校验失败时重试单批；checkpoint 仅记录成功处理的文件

环境变量:
  LLM_API_KEY          必填
  LLM_BASE_URL         默认 https://api.deepseek.com
  LLM_MODEL            默认 deepseek-chat
  LLM_BATCH_LINES      默认 25
  LLM_WORKERS          默认 3
  LLM_REPLACE_EXISTING 设为 1 覆盖已有对照

用法:
  python3 scripts/add_lrc_zh_translations_llm.py
  python3 scripts/add_lrc_zh_translations_llm.py --file lyrics/5000.lrc
  LLM_REPLACE_EXISTING=1 python3 scripts/add_lrc_zh_translations_llm.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:
    from opencc import OpenCC

    _CC = OpenCC("t2s")
except ImportError:
    _CC = None

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "lyrics"
CHECKPOINT = PROJECT / ".lrc_llm_translation_checkpoint.json"
ENV_FILE = PROJECT / ".env"
PLACEHOLDER = "Neko云音乐 暂无歌词"

TS_LINE = re.compile(r"^(\[\d{1,2}:\d{2}(?:[\.:]\d+)?\])(.*)$")
JSON_LINE = re.compile(r'^\s*\{"((?:[^"\\]|\\.)*)"\}\s*$')

SIMILARITY_THRESHOLD = 0.85

METADATA_PREFIXES = (
    "词",
    "曲",
    "编曲",
    "制作人",
    "监制",
    "混音",
    "录音",
    "出品",
    "发行",
    "作词",
    "作曲",
    "原唱",
    "演唱",
    "翻译",
    "Lyrics",
    "Music",
    "Arrange",
    "Produced",
    "Written",
    "Composed",
    "Translator",
)

_NONSENSE_TOKENS = frozenset(
    """
    na la lila lalala balaba bala baraba nanana oh ah uh ooh woo hoo hey yo yeah
    yeh sha boom clap da di do pa pi pu ba tra ra ta ma ha he ho hu ding dong ring
    """.split()
)

SYSTEM_PROMPT = """你是专业歌词翻译，为音乐 App 的 LRC 提供「简体中文对照」行。

要求：
1. 忠实原意，符合歌词语境、情感与语气
2. 只使用简体中文与中文标点、数字、全角空格「　」，不得出现英文字母、日文假名、韩文、繁体字
3. 行数与输入严格一一对应，不合并、不拆分
4. 只输出 JSON：{"lines": ["...", ...]}
5. 以下情况对应项必须返回空字符串 ""：
   - 原文已是中文（无需中译中）
   - 拟声/口号/无意义音节（如 Na lila balaba、Lalala、Oh oh oh）
   - 纯演奏信息、制作人名单、时间轴标注等非歌词内容"""


@dataclass
class ProcessStats:
    translated: int = 0
    skipped_chinese: int = 0
    skipped_sound: int = 0
    skipped_meta: int = 0
    dropped_redundant: int = 0
    dropped_invalid: int = 0
    still_missing: int = 0
    retried: int = 0


_lock = threading.Lock()
_done: set[str] = set()


def load_dotenv() -> None:
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    return v.strip().lower() in ("1", "true", "yes", "on") if v else default


def load_checkpoint() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    try:
        return set(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("done", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_checkpoint() -> None:
    with _lock:
        CHECKPOINT.write_text(
            json.dumps({"done": sorted(_done)}, ensure_ascii=False),
            encoding="utf-8",
        )


def cjk_count(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")


def kana_count(s: str) -> int:
    return sum(1 for c in s if "\u3040" <= c <= "\u30ff")


def hangul_count(s: str) -> int:
    return sum(1 for c in s if "\uac00" <= c <= "\ud7a3" or "\u1100" <= c <= "\u11ff")


def latin_count(s: str) -> int:
    return sum(1 for c in s if c.isalpha() and ord(c) < 0x300)


def meaningful_chars(s: str) -> int:
    return sum(
        1
        for c in s
        if not c.isspace()
        and c not in "，。、；：？！…—·「」『』（）【】[](){}:;,.!?\'\"-"
    )


def is_placeholder_line(text: str) -> bool:
    t = text.strip()
    return bool(re.fullmatch(r"[.…·\s]+", t))


def is_metadata_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if PLACEHOLDER in t:
        return True
    if is_placeholder_line(t):
        return True
    return any(t.startswith(p) for p in METADATA_PREFIXES)


def is_chinese_dominant(s: str) -> bool:
    if kana_count(s) >= 1 or hangul_count(s) >= 1:
        return False
    cjk = cjk_count(s)
    latin = latin_count(s)
    m = meaningful_chars(s)
    if m == 0:
        return False
    if cjk >= 1 and latin == 0:
        return True
    return cjk >= 2 and cjk / m >= 0.4 and latin <= max(2, cjk * 0.15)


def latin_tokens(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", s.lower())


def is_syllable_repeat_token(token: str) -> bool:
    return len(token) >= 4 and bool(
        re.fullmatch(r"(?:la|na|oh|ha|ba|da|di|do|pa|sha|woo|lila|bala)+", token)
    )


def is_untranslatable_sound(s: str) -> bool:
    if cjk_count(s) > 0 or kana_count(s) > 0 or hangul_count(s) > 0:
        return False
    if latin_count(s) < 3:
        return False
    tokens = latin_tokens(s)
    if not tokens:
        return False
    if len(tokens) == 1 and is_syllable_repeat_token(tokens[0]):
        return True
    if len(tokens) <= 12 and all(
        t in _NONSENSE_TOKENS or is_syllable_repeat_token(t) for t in tokens
    ):
        return True
    return False


def normalize_text(s: str) -> str:
    s = re.sub(r"[\s　]+", "", s)
    s = re.sub(r"[，。、；：？！…—·「」『』（）【】\[\](){}:;,.!?\'\"\-]", "", s)
    return s.lower()


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_redundant_zh_pair(orig: str, trans: str) -> bool:
    if not is_chinese_dominant(orig) or not is_chinese_dominant(trans):
        return False
    o, t = normalize_text(orig), normalize_text(trans)
    if not o or not t:
        return False
    return o == t or similarity(o, t) >= SIMILARITY_THRESHOLD


def has_traditional(s: str) -> bool:
    if _CC is None:
        return False
    han = "".join(c for c in s if "\u4e00" <= c <= "\u9fff")
    return bool(han) and _CC.convert(han) != han


def is_pure_simplified_zh(s: str) -> bool:
    if not s.strip():
        return True
    if kana_count(s) or hangul_count(s) or latin_count(s):
        return False
    if re.search(r"[\u0400-\u04FF]", s):
        return False
    if has_traditional(s):
        return False
    allowed = set("，。、；：？！…—·「」『』（）【】《》〈〉""''　()[]*~/@#%&+=<>|^_.\"'-")
    for c in s:
        if c.isspace() or c in allowed:
            continue
        if c.isdigit():
            continue
        if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf":
            continue
        if ord(c) < 128 and not c.isalpha():
            continue
        return False
    return True


def to_simplified(s: str) -> str:
    if _CC is None:
        return s
    return _CC.convert(s)


def classify_skip(lyric: str) -> str | None:
    if is_metadata_line(lyric):
        return "meta"
    if is_chinese_dominant(lyric):
        return "chinese"
    if is_untranslatable_sound(lyric):
        return "sound"
    return None


def sanitize_translation(orig: str, trans: str, stats: ProcessStats) -> str:
    trans = to_simplified(trans.strip())
    if not trans:
        return ""
    if is_placeholder_line(trans):
        stats.dropped_invalid += 1
        return ""
    if classify_skip(orig):
        stats.dropped_invalid += 1
        return ""
    if not is_pure_simplified_zh(trans):
        stats.dropped_invalid += 1
        return ""
    if is_redundant_zh_pair(orig, trans):
        stats.dropped_redundant += 1
        return ""
    return clean_translation_text(trans)


def clean_translation_text(text: str) -> str:
    """去掉误转义，英文双引号改为中文引号，避免写入 {\\\"...\\\"}。"""
    text = text.strip()
    text = text.replace('\\"', '"').replace("\\\\", "\\")
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        inner = text[1:-1].strip()
        if '"' not in inner:
            text = inner
        else:
            break
    text = re.sub(r'"([^"]*)"', r"「\1」", text)
    text = text.replace('"', "")
    return text.strip()


def escape_translation(text: str) -> str:
    text = clean_translation_text(text)
    return text.replace("\\", "\\\\").replace('"', '\\"')


def parse_json_line(line: str) -> str | None:
    m = JSON_LINE.match(line.strip())
    if not m:
        return None
    return clean_translation_text(
        m.group(1).replace('\\"', '"').replace("\\\\", "\\")
    )


def extract_translation_inner(line: str) -> str | None:
    """从对照行解析正文，兼容误转义与 {"{" 损坏格式。"""
    s = line.strip()
    if not s.startswith('{"') or not s.endswith('"}'):
        return None
    inner = s[2:-2]
    if inner.startswith('{"') and inner.endswith('"}'):
        inner = inner[2:-2]
    return clean_translation_text(inner.replace('\\"', '"').replace("\\\\", "\\"))


def format_translation_line(line: str, inner: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    return f'{indent}{{"' + escape_translation(inner) + '"}}'


def fix_escaped_quotes_in_file(path: Path) -> int:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    changed = 0
    out: list[str] = []
    for line in lines:
        inner = extract_translation_inner(line)
        needs_fix = False
        if inner is not None:
            raw = line.strip()[2:-2]
            if raw.startswith('{"') or '\\"' in raw or raw != escape_translation(inner):
                needs_fix = True
        if needs_fix and inner is not None:
            out.append(format_translation_line(line, inner))
            changed += 1
        else:
            out.append(line)
    if changed:
        text = "\n".join(out)
        orig = path.read_text(encoding="utf-8-sig", errors="replace")
        if orig.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
    return changed


def chat_translate(lines: list[str], api_key: str, base_url: str, model: str) -> list[str]:
    if not lines:
        return []
    payload = {
        "model": model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请翻译以下歌词行（保持顺序）：\n"
                + json.dumps(lines, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content).strip()
            data = json.loads(content)
            out = data.get("lines") or data.get("translations") or data.get("result")
            if not isinstance(out, list) or len(out) != len(lines):
                raise ValueError(f"LLM 返回 JSON 格式或行数不正确: {content[:200]}")
            return [str(x) for x in out]
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as e:
            wait = min(60, 2 ** attempt)
            print(f"    LLM 重试第 {attempt + 1} 次: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("LLM 翻译失败：已重试多次仍无响应")


def strip_existing_translations(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if TS_LINE.match(line) and i + 1 < len(lines) and JSON_LINE.match(lines[i + 1].strip()):
            i += 2
            continue
        i += 1
    return out


def find_missing_translations(lines: list[str]) -> list[tuple[int, str]]:
    """需要外语对照但下一行尚无 {"..."} 的行。"""
    missing: list[tuple[int, str]] = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        m = TS_LINE.match(line)
        if not m:
            continue
        lyric = m.group(2).strip()
        if not lyric or classify_skip(lyric):
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if parse_json_line(nxt) is not None:
            skip_next = True
            continue
        missing.append((i, lyric))
    return missing


def collect_pending(lines: list[str], replace_existing: bool) -> list[tuple[int, str]]:
    pending: list[tuple[int, str]] = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        m = TS_LINE.match(line)
        if not m:
            continue
        lyric = m.group(2).strip()
        if not lyric:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        has_tr = parse_json_line(nxt) is not None
        if has_tr and not replace_existing:
            skip_next = True
            continue
        if has_tr and replace_existing:
            skip_next = True
        pending.append((i, lyric))
    return pending


def file_has_missing_translations(path: Path) -> bool:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if PLACEHOLDER in text and PLACEHOLDER == text.strip():
        return False
    return bool(find_missing_translations(text.splitlines()))


def process_file(
    path: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    replace_existing: bool,
) -> tuple[bool, ProcessStats, bool]:
    stats = ProcessStats()
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if PLACEHOLDER in text and PLACEHOLDER == text.strip():
        return False, stats, True

    lines = text.splitlines()
    if replace_existing:
        lines = strip_existing_translations(lines)

    pending = collect_pending(lines, False)
    if not pending:
        return False, stats, True

    need_llm: list[tuple[int, str]] = []
    local_empty: dict[int, str] = {}

    for i, lyric in pending:
        reason = classify_skip(lyric)
        if reason == "meta":
            stats.skipped_meta += 1
            local_empty[i] = ""
        elif reason == "chinese":
            stats.skipped_chinese += 1
            local_empty[i] = ""
        elif reason == "sound":
            stats.skipped_sound += 1
            local_empty[i] = ""
        else:
            need_llm.append((i, lyric))

    translations: dict[int, str] = dict(local_empty)

    for start in range(0, len(need_llm), batch_size):
        chunk = need_llm[start : start + batch_size]
        idxs = [i for i, _ in chunk]
        src = [t for _, t in chunk]
        raw = chat_translate(src, api_key, base_url, model)
        for idx, orig, trans in zip(idxs, src, raw):
            cleaned = sanitize_translation(orig, trans, stats)
            if cleaned:
                stats.translated += 1
            translations[idx] = cleaned
        time.sleep(0.2)

    retry: list[tuple[int, str]] = [
        (i, orig) for i, orig in need_llm if not translations.get(i, "").strip()
    ]
    if retry:
        for start in range(0, len(retry), min(batch_size, 10)):
            chunk = retry[start : start + min(batch_size, 10)]
            idxs = [i for i, _ in chunk]
            src = [t for _, t in chunk]
            raw = chat_translate(src, api_key, base_url, model)
            for idx, orig, trans in zip(idxs, src, raw):
                cleaned = sanitize_translation(orig, trans, stats)
                if cleaned:
                    stats.translated += 1
                    stats.retried += 1
                translations[idx] = cleaned
            time.sleep(0.2)

    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if i in translations and translations[i].strip():
            out.append('{"' + escape_translation(translations[i]) + '"}')

    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8", newline="\n")

    final_lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    stats.still_missing = len(find_missing_translations(final_lines))
    complete = stats.still_missing == 0
    changed = stats.translated > 0
    return changed, stats, complete


def format_file_summary(stats: ProcessStats) -> str:
    if stats.translated:
        msg = f"写入 {stats.translated} 行对照"
        if stats.retried:
            msg += f"（含补翻 {stats.retried} 行）"
        if stats.still_missing:
            msg += f"，仍缺 {stats.still_missing} 行"
        return msg
    parts: list[str] = []
    if stats.skipped_chinese:
        parts.append(f"中文原文 {stats.skipped_chinese} 行")
    if stats.skipped_sound: 
        parts.append(f"拟声 {stats.skipped_sound} 行")
    if stats.skipped_meta:
        parts.append(f"元数据 {stats.skipped_meta} 行")
    if stats.dropped_redundant:
        parts.append(f"丢弃重复 {stats.dropped_redundant} 行")
    if stats.dropped_invalid:
        parts.append(f"丢弃无效 {stats.dropped_invalid} 行")
    if not parts:
        return "无可翻译内容"
    return "无需翻译（" + "，".join(parts) + "）"


def worker(
    path: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    replace_existing: bool,
) -> tuple[str, bool, ProcessStats, bool, str | None]:
    try:
        ok, stats, complete = process_file(
            path, api_key, base_url, model, batch_size, replace_existing
        )
        return path.name, ok, stats, complete, None
    except Exception as e:
        return path.name, False, ProcessStats(), False, str(e)


def main() -> int:
    global _done
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM 批量添加 LRC 中文对照（优化版）")
    parser.add_argument("--file", type=Path, help="只处理单个 lrc 文件")
    parser.add_argument("--replace-existing", action="store_true", help="覆盖已有对照行")
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="补翻已有 checkpoint 但对照不完整的文件",
    )
    parser.add_argument(
        "--fix-escaped-quotes",
        action="store_true",
        help="修复对照行中误加的 \\\" 转义（不改其它内容）",
    )
    args = parser.parse_args()

    if args.fix_escaped_quotes:
        targets = (
            [args.file if args.file.is_absolute() else PROJECT / args.file]
            if args.file
            else sorted(ROOT.glob("*.lrc"))
        )
        total = 0
        for p in targets:
            n = fix_escaped_quotes_in_file(p)
            if n:
                total += n
                print(f"修复 {p.name}: {n} 行", flush=True)
        print(f"共修复 {total} 行", flush=True)
        return 0

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        print(
            "请设置 LLM_API_KEY，例如：\n"
            "  export LLM_API_KEY=sk-...\n"
            "  export LLM_BASE_URL=https://api.deepseek.com\n"
            "  export LLM_MODEL=deepseek-chat",
            file=sys.stderr,
        )
        return 2

    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("LLM_MODEL", "deepseek-chat").strip()
    batch_size = int(os.environ.get("LLM_BATCH_LINES", "25"))
    workers = int(os.environ.get("LLM_WORKERS", "3"))
    replace_existing = args.replace_existing or env_bool("LLM_REPLACE_EXISTING")

    if args.file:
        files = [args.file if args.file.is_absolute() else PROJECT / args.file]
    elif args.retry_incomplete:
        files = [p for p in sorted(ROOT.glob("*.lrc")) if file_has_missing_translations(p)]
    else:
        _done = load_checkpoint()
        files = [p for p in sorted(ROOT.glob("*.lrc")) if p.name not in _done]

    if not files:
        if args.retry_incomplete:
            print("没有对照不完整的文件")
        else:
            print("没有待处理的文件")
        return 0

    opencc_hint = "已启用" if _CC else "未启用（可 pip install opencc-python-reimplemented）"
    print(
        f"开始翻译：共 {len(files)} 个文件，模型={model}，每批={batch_size} 行，"
        f"并发={workers}，繁转简={opencc_hint}",
        flush=True,
    )

    changed = errors = incomplete = 0
    totals = ProcessStats()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                worker, p, api_key, base_url, model, batch_size, replace_existing
            ): p
            for p in files
        }
        for n, fut in enumerate(as_completed(futs), 1):
            name, did, stats, complete, err = fut.result()
            if err:
                errors += 1
                print(f"[{n}/{len(files)}] 错误 {name}: {err}", flush=True)
            else:
                if not args.file and complete:
                    with _lock:
                        _done.add(name)
                elif not args.file and not complete:
                    incomplete += 1
                summary = format_file_summary(stats)
                if stats.translated:
                    changed += 1
                    tag = "完成" if complete else "部分"
                    print(f"[{n}/{len(files)}] {tag} {name}：{summary}", flush=True)
                else:
                    print(f"[{n}/{len(files)}] 跳过 {name}：{summary}", flush=True)
                totals.translated += stats.translated
                totals.skipped_chinese += stats.skipped_chinese
                totals.skipped_sound += stats.skipped_sound
                totals.skipped_meta += stats.skipped_meta
                totals.dropped_redundant += stats.dropped_redundant
                totals.dropped_invalid += stats.dropped_invalid
                totals.still_missing += stats.still_missing
                totals.retried += stats.retried
            if not args.file and n % 20 == 0:
                save_checkpoint()

    if not args.file:
        save_checkpoint()

    print(
        f"全部结束：已写入对照 {changed} 个文件，"
        f"不完整 {incomplete} 个，跳过 {len(files) - changed - errors - incomplete} 个，失败 {errors} 个",
        flush=True,
    )
    print(
        f"统计：写入对照 {totals.translated} 行 | "
        f"仍缺对照 {totals.still_missing} 行 | "
        f"跳过中文原文 {totals.skipped_chinese} 行，"
        f"跳过拟声 {totals.skipped_sound} 行，"
        f"跳过元数据 {totals.skipped_meta} 行，"
        f"丢弃重复 {totals.dropped_redundant} 行，"
        f"丢弃无效 {totals.dropped_invalid} 行",
        flush=True,
    )
    if incomplete:
        print(
            "提示：不完整文件未写入 checkpoint，请用 --retry-incomplete 继续补翻。",
            flush=True,
        )
    if changed == 0 and errors == 0 and len(files) <= 10:
        print(
            "提示：若本批文件均为中文歌词或「暂无歌词」占位，则不会调用 LLM，属正常情况。",
            flush=True,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
