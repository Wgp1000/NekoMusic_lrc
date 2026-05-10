#!/usr/bin/env python3
"""
为 music 目录下所有 MP3 / FLAC / WAV（不递归子目录）写入统一的「注释」广告文案。

- MP3：ID3 COMM（UTF-8），并删除已有 COMM；ORGANIZATION（TXXX）、PUBLISHER（TPUB）。
- FLAC：Vorbis COMMENT、ORGANIZATION、PUBLISHER。
- WAV：若存在 ffmpeg，先无损写入 RIFF 侧 comment（部分播放器只读这一层），再写 ID3 COMM；
  否则仅写 ID3。保存前 chmod 增加本用户写权限；权限仍失败时用临时文件再 os.replace。
- 若 .mp3 / .flac 实际为 MP4，则回退写入 MP4 的 ©cmt。
- 嵌入歌词（非外部 .lrc）：在已有内嵌歌词最前插入两行 LRC 轴文案；无歌词则仅写入该两行。
  MP3/WAV 用 ID3 USLT；FLAC 用 Vorbis「lyrics」；MP4 用 ©lyr。已以相同首行开头则跳过。

在仓库根目录执行：python3 set_comment_ad.py
"""
from __future__ import annotations

import errno
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from mutagen.flac import FLAC, FLACNoHeaderError
from mutagen.id3 import COMM, TPUB, TXXX, USLT
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.mp4 import MP4, MP4FreeForm
from mutagen.wave import WAVE

COMMENT = (
    "更多免费无损音乐就来Neko云音乐 https://music.cnmsb.xin "
    "For more free lossless music, visit Neko Cloud Music: https://music.cnmsb.xin"
)

ORGANIZATION = "Neko Music"
PUBLISHER = "music.cnmsb.xin"

# 内嵌歌词（LRC 时间轴）横幅，写在已有歌词最前
LYRICS_BANNER = (
    "[00:00.00]资源来自Neko云音乐 Resources from Neko Cloud Music\n"
    "\n"
    "[00:00.10]获取更多无损音乐https://music.cnmsb.xin/ Get more lossless music at https://music.cnmsb.xin/\n"
)
LYRICS_BANNER_FIRST = "[00:00.00]资源来自Neko云音乐 Resources from Neko Cloud Music"

# 脚本在仓库根目录，音频在子目录 music/
AUDIO_DIR = Path(__file__).resolve().parent / "music"
MP4_COMMENT = "\xa9cmt"
MP4_LYRICS = "\xa9lyr"
# iTunes 自由格式，便于与部分工具中的 ORGANIZATION / PUBLISHER 名称对齐
MP4_FF_ORGANIZATION = "----:com.apple.iTunes:ORGANIZATION"
MP4_FF_PUBLISHER = "----:com.apple.iTunes:PUBLISHER"


def _ensure_user_writable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        os.chmod(path, mode | stat.S_IWUSR)
    except OSError:
        pass


def _strip_comm_id3(tags) -> None:
    for key in list(tags.keys()):
        if key.startswith("COMM"):
            del tags[key]


def _set_id3_organization_publisher(tags) -> None:
    """ID3：发行方 ORGANIZATION（TXXX）、出版者 PUBLISHER（TPUB 标准帧）。"""
    for key in list(tags.keys()):
        if key == "TPUB":
            del tags[key]
        elif key.startswith("TXXX:"):
            frame = tags[key]
            if getattr(frame, "desc", "") == "ORGANIZATION":
                del tags[key]
    tags.add(TPUB(encoding=3, text=PUBLISHER))
    tags.add(TXXX(encoding=3, desc="ORGANIZATION", text=ORGANIZATION))


def _prepend_embedded_lyrics_id3(tags) -> None:
    """在 ID3 内嵌歌词（USLT）最前插入 LYRICS_BANNER；无 USLT 则新建。"""
    uslt_keys = [k for k in list(tags.keys()) if k.startswith("USLT")]
    if not uslt_keys:
        tags.add(
            USLT(encoding=3, lang="zho", desc="", text=LYRICS_BANNER.rstrip("\n"))
        )
        return
    parts: list[str] = []
    lang = "zho"
    for k in sorted(uslt_keys):
        fr = tags[k]
        parts.append(getattr(fr, "text", "") or "")
        if getattr(fr, "lang", None):
            lang = fr.lang or lang
    body = "\n\n".join(p for p in parts if p.strip())
    if body.strip():
        first = body.lstrip("\ufeff").splitlines()[0]
        if first.strip() == LYRICS_BANNER_FIRST.strip():
            return
        new_text = (LYRICS_BANNER + body).rstrip("\n")
    else:
        new_text = LYRICS_BANNER.rstrip("\n")
    for k in uslt_keys:
        del tags[k]
    tags.add(USLT(encoding=3, lang=lang, desc="", text=new_text))


def _prepend_flac_embedded_lyrics(audio: FLAC) -> None:
    body: str | None = None
    for key in ("lyrics", "LYRICS"):
        if key in audio:
            body = "\n\n".join(audio[key])
            break
    if body is None:
        audio["lyrics"] = [LYRICS_BANNER.rstrip("\n")]
        return
    if body.strip():
        lines = body.lstrip("\ufeff").splitlines()
        first = lines[0] if lines else ""
        if first.strip() == LYRICS_BANNER_FIRST.strip():
            return
    audio["lyrics"] = [(LYRICS_BANNER + body).rstrip("\n")]


def _prepend_mp4_embedded_lyrics(audio: MP4) -> None:
    k = MP4_LYRICS
    if k not in audio or not audio[k]:
        audio[k] = [LYRICS_BANNER.rstrip("\n")]
        return
    cur = "\n".join(audio[k])
    if cur.strip():
        lines = cur.lstrip("\ufeff").splitlines()
        first = lines[0] if lines else ""
        if first.strip() == LYRICS_BANNER_FIRST.strip():
            return
    audio[k] = [(LYRICS_BANNER + cur).rstrip("\n")]


def patch_mp3(path: Path) -> None:
    audio = MP3(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    _set_id3_organization_publisher(audio.tags)
    _prepend_embedded_lyrics_id3(audio.tags)
    audio.save()


def patch_flac(path: Path) -> None:
    audio = FLAC(str(path))
    audio["comment"] = [COMMENT]
    audio["ORGANIZATION"] = [ORGANIZATION]
    audio["PUBLISHER"] = [PUBLISHER]
    _prepend_flac_embedded_lyrics(audio)
    audio.save()


def patch_mp4(path: Path) -> None:
    audio = MP4(str(path))
    audio[MP4_COMMENT] = [COMMENT]
    audio[MP4_FF_ORGANIZATION] = [MP4FreeForm(ORGANIZATION.encode("utf-8"))]
    audio[MP4_FF_PUBLISHER] = [MP4FreeForm(PUBLISHER.encode("utf-8"))]
    _prepend_mp4_embedded_lyrics(audio)
    audio.save()


def _wav_apply_id3_comm(path: Path) -> None:
    audio = WAVE(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    _set_id3_organization_publisher(audio.tags)
    _prepend_embedded_lyrics_id3(audio.tags)
    try:
        audio.save()
    except PermissionError:
        _wav_apply_id3_comm_via_temp(path)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EPERM):
            _wav_apply_id3_comm_via_temp(path)
        else:
            raise


def _wav_apply_id3_comm_via_temp(path: Path) -> None:
    fd, raw = tempfile.mkstemp(
        suffix=".wav", prefix=f".{path.stem}.", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(raw)
    try:
        shutil.copy2(path, tmp)
        audio = WAVE(str(tmp))
        if audio.tags is None:
            audio.add_tags()
        _strip_comm_id3(audio.tags)
        audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
        _set_id3_organization_publisher(audio.tags)
        _prepend_embedded_lyrics_id3(audio.tags)
        audio.save()
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _patch_wav_ffmpeg_riff_then_id3(path: Path) -> None:
    """先 ffmpeg 写 RIFF comment，再 mutagen 写 ID3 COMM，最后原子替换。"""
    tmp = path.parent / f".{path.name}.ff.{os.getpid()}.wav"
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:a",
                "copy",
                "-metadata",
                f"comment={COMMENT}",
                str(tmp),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip() or f"退出码 {r.returncode}"
            raise RuntimeError(msg[:800])
        _wav_apply_id3_comm(tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def patch_wav(path: Path) -> None:
    _ensure_user_writable(path)
    if shutil.which("ffmpeg"):
        try:
            _patch_wav_ffmpeg_riff_then_id3(path)
            return
        except Exception:
            pass
    _wav_apply_id3_comm(path)


def process_file(path: Path) -> None:
    suf = path.suffix.lower()
    if suf == ".mp3":
        try:
            patch_mp3(path)
        except HeaderNotFoundError:
            patch_mp4(path)
    elif suf == ".flac":
        try:
            patch_flac(path)
        except FLACNoHeaderError:
            patch_mp4(path)
    elif suf in (".wav", ".wave"):
        patch_wav(path)
    else:
        raise ValueError(f"不支持的扩展名：{suf}")


def main() -> int:
    if not AUDIO_DIR.is_dir():
        print(f"错误：找不到音频目录 {AUDIO_DIR}", file=sys.stderr)
        return 1

    ok = err = 0
    exts = {".mp3", ".flac", ".wav", ".wave"}
    for path in sorted(AUDIO_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        try:
            process_file(path)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"失败：{path}\n原因：{e}", file=sys.stderr)
            err += 1

    print(f"完成：成功 {ok} 个，失败 {err} 个。")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
