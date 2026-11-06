"""음성 인식 모델: model.bin까지 있어야 '받음', 미리 받기는 받는 폴더 크기로 진행률을 알린다 (인터넷 없이 가짜 받기로)."""

import time

import pytest

pytest.importorskip("faster_whisper")

from ks_studio.services import speech  # noqa: E402


def snapshot(folder):
    return folder / f"models--Systran--faster-whisper-{speech.MODEL}" / "snapshots" / "abc"


def test_ready_needs_model_bin(tmp_path):
    assert not speech.speech_model_ready(tmp_path)
    snapshot(tmp_path).mkdir(parents=True)
    (snapshot(tmp_path) / "config.json").write_text("{}")
    assert not speech.speech_model_ready(tmp_path)  # 받다 끊긴 폴더
    (snapshot(tmp_path) / "model.bin").write_bytes(b"x")
    assert speech.speech_model_ready(tmp_path)


def test_download_reports_progress_and_finishes(tmp_path, monkeypatch):
    import faster_whisper.utils

    def fake_download(size, cache_dir):
        assert size == speech.MODEL and cache_dir == str(tmp_path)
        snapshot(tmp_path).mkdir(parents=True)
        (snapshot(tmp_path) / "model.bin").write_bytes(b"x" * 1000)
        time.sleep(0.7)  # 진행률 재기(0.5초마다)가 한 번은 돌게

    monkeypatch.setattr(faster_whisper.utils, "download_model", fake_download)
    seen = []
    speech.download_speech_model(seen.append, tmp_path)
    assert speech.speech_model_ready(tmp_path)
    assert len(seen) >= 2 and 0 < seen[0] < 1 and seen[-1] == 1.0
