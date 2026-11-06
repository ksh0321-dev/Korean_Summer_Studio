"""음성 → 자막 (faster-whisper large-v3, GPU). 한국어로 말해도 영어 자막으로 옮길 수 있다 (task="translate").

- 모델은 처음 쓸 때 프로젝트 models/whisper/에 내려받는다 (약 3GB, 저장소에 올리지 않음)
- NVIDIA GPU가 없으면 CPU(int8)로 — 느리지만 된다
- 실측 (RTX 5080): 모델 열기 3초(내려받은 뒤), 갤럭시 13초 영상 번역 0.8초
- 음악만 나오는 구간에서 없는 말을 지어내는 일이 있다 (Whisper의 알려진 버릇) → 말소리 구간만(VAD) 듣고,
  앞 문장에 이어 짓지 않게 하고, 영상 길이 밖·같은 문장 반복은 버린다. 그래도 남으면 표에서 지운다
"""

import logging
import os
import queue
import threading
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.app.paths import MODELS_DIR
from ks_studio.core.subtitles import Cue

log = logging.getLogger(__name__)

MODEL = "large-v3"  # turbo는 번역(translate)을 배우지 않아 영어 자막에 못 쓴다
MODEL_DIR = MODELS_DIR / "whisper"
SAMPLE_RATE = 16_000


def speech_unavailable_reason() -> str | None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as e:
        return f"음성 인식 패키지가 없습니다 ({e.name}). 설치 방법은 pyproject.toml의 [ai] 설명을 보세요"
    return None


def load_audio(path: Path) -> np.ndarray:
    """16kHz 모노 float32 (-1~1). faster-whisper의 내장 디코더는 PyAV 19에서 깨져서 직접 푼다.

    s16으로 리샘플해서 바꾼다 — 실측: 이 PyAV에서 'flt'(실수)로 리샘플하면 값이 전부 0이었다.
    """
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, np.float32)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32) / 32768 if chunks else np.zeros(0, np.float32)


def resample_48k(audio: np.ndarray) -> np.ndarray:
    """48kHz → 16kHz (음성 인식 입력)."""
    from scipy.signal import resample_poly

    return resample_poly(audio.astype(np.float32), 1, 3).astype(np.float32)


def clean(segments, duration: float) -> list[Cue]:
    """faster-whisper 구간 → 자막 줄. 빈 글, 영상 끝 뒤, 바로 앞과 같은 문장(지어낸 반복)은 버린다."""
    cues: list[Cue] = []
    for start, end, text in segments:
        text = text.strip()
        if not text or start >= duration:
            continue
        if cues and cues[-1].text == text:
            continue
        cues.append(Cue(round(start, 2), round(min(end, duration), 2), text))
    return cues


class SpeechWorker(QObject):
    """음성 인식을 스레드 하나에서 (모델은 처음 쓸 때 연다)."""

    status = Signal(Path, str)  # 영상, 지금 하는 일
    progress = Signal(Path, float)  # 영상, 0~1
    done = Signal(Path, object)  # 영상, list[Cue]
    failed = Signal(Path, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = None
        self._jobs: queue.Queue = queue.Queue()
        self.busy_with: Path | None = None
        threading.Thread(target=self._worker, daemon=True, name="speech").start()

    def request(self, path: Path, task: str, audio_fn=None) -> None:
        """path의 소리를 알아듣는다. audio_fn을 주면 그 함수가 돌려준 소리(48kHz 모노 float32)로 (컷편집 타임라인)."""
        self.busy_with = path
        self._jobs.put((path, task, audio_fn))

    def _load(self, path: Path):
        if self._model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows 심볼릭 링크 경고 (동작엔 상관없음)
            from faster_whisper import WhisperModel

            done = (MODEL_DIR / f"models--Systran--faster-whisper-{MODEL}").exists()
            self.status.emit(path, "음성 인식 모델 여는 중…" if done else "음성 인식 모델 내려받는 중 (처음 한 번, 약 3GB)…")
            try:
                self._model = WhisperModel(MODEL, device="cuda", compute_type="float16", download_root=str(MODEL_DIR))
            except Exception as e:  # NVIDIA GPU가 없거나 CUDA를 못 쓰면
                log.info("음성 인식 GPU를 못 써서 CPU로: %s", e)
                self._model = WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=str(MODEL_DIR))
        return self._model

    def _worker(self) -> None:
        while True:
            path, task, audio_fn = self._jobs.get()
            try:
                model = self._load(path)
                self.status.emit(path, "소리 읽는 중…")
                audio = load_audio(path) if audio_fn is None else resample_48k(audio_fn())
                duration = len(audio) / SAMPLE_RATE
                if duration == 0:
                    raise RuntimeError("소리가 없는 영상입니다")
                self.status.emit(path, "말 알아듣는 중…")
                segments, _info = model.transcribe(audio, task=task, vad_filter=True, condition_on_previous_text=False)
                found = []
                for segment in segments:  # 하나씩 나오므로 진행률을 보여 줄 수 있다
                    found.append((segment.start, segment.end, segment.text))
                    self.progress.emit(path, min(1.0, segment.end / duration))
                self.busy_with = None
                self.done.emit(path, clean(found, duration))
            except Exception as e:
                log.exception("음성 인식 실패: %s", path)
                self.busy_with = None
                self.failed.emit(path, str(e))
