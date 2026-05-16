#!/usr/bin/env python3
"""
用大模型为 LRC 添加 {"中文对照"} 行（OpenAI 兼容 API：DeepSeek / OpenAI / 通义 / Moonshot 等）。

环境变量（必填其一）:
  LLM_API_KEY          API Key
  LLM_BASE_URL         默认 https://api.deepseek.com
  LLM_MODEL            默认 deepseek-chat

可选:
  LLM_BATCH_LINES      每次请求翻译行数，默认 25
  LLM_WORKERS          并发文件数，默认 3
  LLM_REPLACE_EXISTING 设为 1 时覆盖已有 {"..."} 对照（用于替换机翻）

用法:
  export LLM_API_KEY=sk-...
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
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "lyrics"
CHECKPOINT = PROJECT / ".lrc_llm_translation_checkpoint.json"
ENV_FILE = PROJECT / ".env"
PLACEHOLDER = "Neko云音乐 暂无歌词"

TS_LINE = re.compile(r"^(\[\d{1,2}:\d{2}(?:[\.:]\d+)?\])(.*)$")
JSON_LINE = re.compile(r"^\s*\{.*\}\s*$")

SYSTEM_PROMPT = """你是专业歌词翻译，为音乐 App 的 LRC 歌词提供中文对照行。

要求：
1. 忠实原意，符合歌词语境、情感和语气（口语/诗意/动漫台词等）
2. 中文自然流畅，避免机翻腔和生硬直译
3. 保留专有名词、乐队名、角色名等可音译或业界通用译名
4. 行数与输入严格一一对应，不要合并或拆分
5. 可适当用全角空格「　」分隔意群，与常见 LRC 对照风格一致
6. 只输出 JSON，格式为 {"lines": ["翻译1", "翻译2", ...]}，不要其它说明
7. 如果原文本身就是中文，对应项返回空字符串 ""，不要重复翻译"""

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


def escape_translation(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def chat_translate(lines: list[str], api_key: str, base_url: str, model: str) -> list[str]:
    if not lines:
        return []
    payload = {
        "model": model,
        "temperature": 0.3,
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
                raise ValueError(f"bad LLM JSON shape or length: {content[:200]}")
            return [str(x) for x in out]
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as e:
            wait = min(60, 2 ** attempt)
            print(f"    LLM retry {attempt + 1}: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("LLM translation failed after retries")


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
        has_tr = bool(JSON_LINE.match(nxt))
        if has_tr and not replace_existing:
            skip_next = True
            continue
        if has_tr and replace_existing:
            skip_next = True
        pending.append((i, lyric))
    return pending


def process_file(
    path: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    replace_existing: bool,
) -> bool:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if PLACEHOLDER in text:
        return False

    lines = text.splitlines()
    pending = collect_pending(lines, replace_existing)
    if not pending:
        return False

    # 若覆盖模式，先去掉旧的对照行
    if replace_existing:
        new_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            if TS_LINE.match(line) and i + 1 < len(lines) and JSON_LINE.match(lines[i + 1].strip()):
                i += 2
                continue
            i += 1
        lines = new_lines
        pending = collect_pending(lines, False)

    translations: dict[int, str] = {}
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        idxs = [i for i, _ in chunk]
        src = [t for _, t in chunk]
        zh = chat_translate(src, api_key, base_url, model)
        for i, t in zip(idxs, zh):
            translations[i] = t
        time.sleep(0.2)

    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if i in translations and translations[i].strip():
            out.append('{"' + escape_translation(translations[i]) + '"}')

    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8", newline="\n")
    return True


def worker(
    path: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    replace_existing: bool,
) -> tuple[str, bool, str | None]:
    try:
        ok = process_file(path, api_key, base_url, model, batch_size, replace_existing)
        return path.name, ok, None
    except Exception as e:
        return path.name, False, str(e)


def main() -> int:
    global _done
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM 批量添加 LRC 中文对照")
    parser.add_argument("--file", type=Path, help="只处理单个 lrc 文件")
    parser.add_argument("--replace-existing", action="store_true", help="覆盖已有对照行")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        print(
            "请设置 LLM_API_KEY，例如：\n"
            "  export LLM_API_KEY=sk-...\n"
            "  export LLM_BASE_URL=https://api.deepseek.com   # 可选\n"
            "  export LLM_MODEL=deepseek-chat                   # 可选",
            file=sys.stderr,
        )
        return 2

    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("LLM_MODEL", "deepseek-chat").strip()
    batch_size = int(os.environ.get("LLM_BATCH_LINES", "25"))
    workers = int(os.environ.get("LLM_WORKERS", "3"))
    replace_existing = args.replace_existing or env_bool("LLM_REPLACE_EXISTING")

    if args.file:
        files = [args.file if args.file.is_absolute() else ROOT.parent / args.file]
    else:
        _done = load_checkpoint()
        files = [p for p in sorted(ROOT.glob("*.lrc")) if p.name not in _done]

    if not files:
        print("nothing to do")
        return 0

    print(
        f"LLM translate: {len(files)} files, model={model}, batch={batch_size}, workers={workers}",
        flush=True,
    )
    changed = errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                worker, p, api_key, base_url, model, batch_size, replace_existing
            ): p
            for p in files
        }
        for n, fut in enumerate(as_completed(futs), 1):
            name, did, err = fut.result()
            if not args.file:
                with _lock:
                    _done.add(name)
            if err:
                errors += 1
                print(f"[{n}/{len(files)}] ERR {name}: {err}", flush=True)
            elif did:
                changed += 1
                print(f"[{n}/{len(files)}] + {name}", flush=True)
            if not args.file and n % 20 == 0:
                save_checkpoint()

    if not args.file:
        save_checkpoint()
    print(f"done changed={changed} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
