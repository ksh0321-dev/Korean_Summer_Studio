"""운영체제마다 고르는 것: AI 실행 장치(CUDA·Core ML·CPU), 인코더 후보, 맥 음성 인식(mlx-whisper) — 맥이 없어도 흉내로."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from ks_studio.services import accel, speech  # noqa: E402


def test_providers_follow_what_is_available(monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    assert accel.onnx_providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]  # 맥
    assert accel.onnx_providers(coreml=False) == ["CPUExecutionProvider"]  # 크기가 바뀌는 모델은 CPU
    assert accel.device_name(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == "mps"
    monkeypatch.setattr(onnxruntime, "get_available_providers",
                        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    assert accel.onnx_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]  # NVIDIA (TensorRT는 안 쓴다)
    assert accel.device_name(accel.onnx_providers()) == "cuda"


def test_session_falls_back_to_cpu_when_accelerator_cannot_open(monkeypatch, tmp_path):
    import onnxruntime

    opened = []

    def fake_session(path, providers):
        opened.append(providers)
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("이 모델은 Core ML이 못 연다")
        return "cpu-session"

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
    assert accel.onnx_session(tmp_path / "m.onnx") == "cpu-session"
    assert opened == [["CoreMLExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]


def test_unavailable_encoders_are_skipped():
    from ks_studio.services.optimize import video_encoder

    name, _ = video_encoder((640, 360), 30, (("nonexistent_encoder", {}), *accel.hardware_encoders("h264", "4M"),
                                             ("libx264", {"preset": "ultrafast"})))
    assert name in ("h264_videotoolbox", "libx264")  # 윈도: videotoolbox가 없어 libx264, 맥: videotoolbox


def test_mac_speech_uses_mlx_whisper(monkeypatch, tmp_path):
    """맥 + mlx-whisper: faster-whisper처럼 (구간들, info)로 돌려준다 — 단어 시간·힌트(initial_prompt)까지."""
    calls = {}

    def transcribe(audio, **kwargs):
        calls.update(kwargs)
        return {"language": "ko", "segments": [{"start": 0.0, "end": 1.0, "text": " 안녕",
                                                 "words": [{"start": 0.1, "end": 0.9, "word": " 안녕", "probability": 0.9}]}]}

    import huggingface_hub  # 플랫폼을 바꾸기 전에 (맥 전용 모듈을 찾으려 든다)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, cache_dir: str(tmp_path / repo))
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "mlx_whisper" else None)
    assert speech.use_mlx() and speech.speech_unavailable_reason() is None
    model = speech.MlxWhisper(tmp_path)
    segments, info = model.transcribe(np.zeros(16000, np.float32), task="transcribe", hotwords="곶감", word_timestamps=True,
                                      vad_filter=True)
    assert info.language == "ko" and segments[0].words[0].word == " 안녕" and segments[0].words[0].end == 0.9
    assert calls["initial_prompt"] == "곶감" and calls["word_timestamps"] and calls["path_or_hf_repo"].endswith("whisper-large-v3-mlx")
    assert speech._repo(tmp_path).name == "models--mlx-community--whisper-large-v3-mlx"
