#!/usr/bin/env python3
"""
为 music 目录下所有 MP3 / FLAC / WAV（不递归子目录）写入统一的「注释」广告文案。

- MP3：ID3 COMM（UTF-8），并删除已有 COMM。
- FLAC：Vorbis COMMENT。
- WAV：RIFF 内嵌 id3 块，COMM 规则与 MP3 相同；若无 id3 则创建。
- 若 .mp3 / .flac 实际为 MP4，则回退写入 MP4 的 ©cmt。

在仓库根目录执行：python3 set_comment_ad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from mutagen.flac import FLAC, FLACNoHeaderError
from mutagen.id3 import COMM
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

COMMENT = (
    "更多免费无损音乐就来Neko云音乐 https://music.cnmsb.xin "
    "For more free lossless music, visit Neko Cloud Music: https://music.cnmsb.xin"
)

# 脚本在仓库根目录，音频在子目录 music/
AUDIO_DIR = Path(__file__).resolve().parent / "music"
MP4_COMMENT = "\xa9cmt"


def _strip_comm_id3(tags) -> None:
    for key in list(tags.keys()):
        if key.startswith("COMM"):
            del tags[key]


def patch_mp3(path: Path) -> None:
    audio = MP3(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    audio.save()


def patch_flac(path: Path) -> None:
    audio = FLAC(str(path))
    audio["comment"] = [COMMENT]
    audio.save()


def patch_mp4(path: Path) -> None:
    audio = MP4(str(path))
    audio[MP4_COMMENT] = [COMMENT]
    audio.save()


def patch_wav(path: Path) -> None:
    audio = WAVE(str(path))
    if audio.tags is None:
        audio.add_tags()
    _strip_comm_id3(audio.tags)
    audio.tags.add(COMM(encoding=3, lang="zho", desc="", text=COMMENT))
    audio.save()


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
