#!/usr/bin/env python3
"""
为 music 目录下所有 MP3 / FLAC / WAV（不递归子目录）写入统一的「注释」广告文案。

- MP3：ID3 COMM（UTF-8），并删除已有 COMM；ORGANIZATION（TXXX）、PUBLISHER（TPUB）。
- FLAC：Vorbis COMMENT、ORGANIZATION、PUBLISHER。
- WAV：若存在 ffmpeg，先无损写入 RIFF 侧 comment（部分播放器只读这一层），再写 ID3 COMM；
  否则仅写 ID3。保存前 chmod 增加本用户写权限；权限仍失败时用临时文件再 os.replace。
- 若 .mp3 / .flac 实际为 MP4，则回退写入 MP4 的 ©cmt。

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
from mutagen.id3 import COMM, TPUB, TXXX
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.mp4 import MP4, MP4FreeForm
from mutagen.wave import WAVE

COMMENT = (
    "更多免费无损音乐就来Neko云音乐 https://music.cnmsb.xin "
    "For more free lossless music, visit Neko Cloud Music: https://music.cnmsb.xin"
)

ORGANIZATION = "Neko Music"
PUBLISHER = "music.cnmsb.xin"

# 脚本在仓库根目录，音频在子目录 music/
AUDIO_DIR = Path(__file__).resolve().parent / "music"
MP4_COMMENT = "\xa9cmt"
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


def patch_mp3(path: Path) -> None:
    audio = MP3(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    _set_id3_organization_publisher(audio.tags)
    audio.save()


def patch_flac(path: Path) -> None:
    audio = FLAC(str(path))
    audio["comment"] = [COMMENT]
    audio["ORGANIZATION"] = [ORGANIZATION]
    audio["PUBLISHER"] = [PUBLISHER]
    audio.save()


def patch_mp4(path: Path) -> None:
    audio = MP4(str(path))
    audio[MP4_COMMENT] = [COMMENT]
    audio[MP4_FF_ORGANIZATION] = [MP4FreeForm(ORGANIZATION.encode("utf-8"))]
    audio[MP4_FF_PUBLISHER] = [MP4FreeForm(PUBLISHER.encode("utf-8"))]
    audio.save()


def _wav_apply_id3_comm(path: Path) -> None:
    audio = WAVE(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    _set_id3_organization_publisher(audio.tags)
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
