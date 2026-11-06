"""프로젝트 정리: 원본은 그대로, 쓰지 않는 사본·캐시·예전 형식·쓰다 만 파일만 지운다."""

import os
import time

from ks_studio.core.workspace import Workspace
from ks_studio.services.cleanup import derived_files, sweep
from ks_studio.services.sources import cache_name, optimized_path


def touch(path, data=b"x", age=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if age:
        old = time.time() - age
        os.utime(path, (old, old))
    return path


def test_sweep_keeps_live_and_removes_the_rest(tmp_path):
    ws = Workspace(tmp_path)
    live = touch(ws.recordings_dir / "a.mp4")
    gone = touch(ws.media_dir / "gone.mp4")  # 프로젝트에서 뺀 영상 (원본은 남아야 한다)
    cache = ws.cache_dir
    key, old_key = cache_name(live), cache_name(gone)
    copy = touch(optimized_path(cache, live))
    copy_key = cache_name(copy)
    keep = [
        touch(cache / f"{key}.audio.npy"),
        touch(cache / f"{copy_key}.thumbs2.npz"),
        touch(cache / "ai_samples" / f"{copy_key}.v1.npz"),
        touch(cache / f"{copy_key}.faces_abc.npy"),
        touch(ws.ai_dir / "fresh.mp4.part"),  # 지금 쓰는 중일 수 있다
    ]
    doomed = [
        touch(cache / f"{key}.thumbs.npz"),  # 예전 형식
        touch(cache / f"{old_key}.peaks.npy"),  # 안 쓰는 영상
        touch(optimized_path(cache, gone)),
        touch(cache / f"{copy_key}.faces_old.npy"),  # 예전에 고른 사람
        touch(cache / "ai_clips" / f"{old_key}.mp4"),
        touch(cache / "export_preview.mp4"),
        touch(ws.final_dir / "x.mp4.part", age=600),
        touch(tmp_path / "project.tmp", age=600),
    ]
    user_file = touch(tmp_path / "notes.tmp", age=600)  # 프로젝트 폴더 바로 아래 사용자 파일은 건드리지 않는다
    freed = sweep(ws, [live], keep=[cache / f"{copy_key}.faces_abc.npy"])
    assert all(f.exists() for f in keep) and live.exists() and gone.exists() and copy.exists() and user_file.exists()
    assert not any(f.exists() for f in doomed)
    assert freed == len(doomed)


def test_derived_files_never_include_the_original(tmp_path):
    ws = Workspace(tmp_path)
    original = touch(ws.media_dir / "phone.mov")
    copy = touch(optimized_path(ws.cache_dir, original))
    files = {
        touch(ws.cache_dir / f"{cache_name(original)}.thumbs2.npz"),
        touch(ws.cache_dir / f"{cache_name(copy)}.audio.npy"),
        touch(ws.cache_dir / "ai_clips" / f"{cache_name(copy)}.mp4"),
        copy,
    }
    touch(ws.cache_dir / "other_0123456789abcdef.audio.npy")
    assert set(derived_files(ws.cache_dir, original)) == files and original not in files
