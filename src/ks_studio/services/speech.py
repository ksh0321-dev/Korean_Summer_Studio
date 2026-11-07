"""음성 → 자막 (faster-whisper large-v3, GPU). 한국어로 말해도 영어 자막으로 옮길 수 있다 (task="translate").

- 단어별 시간까지 받아서 자막 줄을 짧게 나눈다 (core.subtitles.split_words — 숏폼 세로 화면에 1~2줄)
- 두 줄 자막(task="bilingual"): 한국어로 받아 나눈 뒤, 줄마다 그 구간 소리만 다시 '영어로' 돌린다 → 시간이 딱 맞는 번역.
  문장 통째로 번역한 것과 견줘도 뜻은 비슷했다 (실측). 다만 Whisper 번역은 고유명사·음식 이름에 약하다 (곶감 → piece of cake)

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
from ks_studio.core.glossary import speech_hint, translation_hint
from ks_studio.core.subtitles import Cue, Word, split_words

log = logging.getLogger(__name__)

MODEL = "large-v3"  # turbo는 번역(translate)을 배우지 않아 영어 자막에 못 쓴다
MODEL_DIR = MODELS_DIR / "whisper"
SPEECH_MODEL_MB = 3100  # 내려받는 크기 (안내·진행률용)
SAMPLE_RATE = 16_000


def _repo(folder: Path) -> Path:
    return folder / f"models--Systran--faster-whisper-{MODEL}"


def speech_model_ready(folder: Path = MODEL_DIR) -> bool:
    """모델 본체(model.bin)까지 받았나 — 폴더만 보면 받다 끊긴 것도 '있음'이 된다."""
    return any(f.stat().st_size > 0 for f in _repo(folder).glob("snapshots/*/model.bin"))


def download_speech_model(on_progress=None, folder: Path = MODEL_DIR) -> None:
    """음성 인식 모델을 미리 받아 둔다 (자막 만들기를 처음 누를 때 3GB를 기다리지 않게). 이어받기가 된다.

    받는 쪽(huggingface_hub)이 진행률을 알려 주지 않아서, 받는 폴더 크기를 0.5초마다 재서 알린다.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from faster_whisper.utils import download_model

    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            size = 0
            for f in _repo(folder).rglob("*"):
                try:
                    size += f.stat().st_size if f.is_file() else 0
                except OSError:  # 받는 중에 이름이 바뀐 임시 파일
                    pass
            on_progress(min(0.99, size / (SPEECH_MODEL_MB * 1e6)))

    watcher = threading.Thread(target=watch, daemon=True, name="speech-model-size") if on_progress else None
    if watcher:
        watcher.start()
    try:
        download_model(MODEL, cache_dir=str(folder))
    finally:
        stop.set()
    if on_progress:
        on_progress(1.0)


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


TRANSLATE_PAD = 0.2  # 줄을 번역할 때 앞뒤로 이만큼 소리를 더 준다 (첫·끝 음절이 잘리지 않게)


def segment_words(segments, duration: float) -> list[Word]:
    """faster-whisper 구간 [(시작, 끝, 글, [(시작, 끝, 단어), ...]), ...] → 단어들.

    빈 글, 영상 끝 뒤, 바로 앞과 같은 문장(음악 구간에서 지어낸 반복)은 버린다. 단어 시간이 없는 구간은 통째로 한 단어.
    """
    words: list[Word] = []
    previous = None
    for start, end, text, seg_words in segments:
        text = text.strip()
        if not text or start >= duration or text == previous:
            continue
        previous = text
        found = [Word(round(a, 3), round(min(b, duration), 3), w) for a, b, w in seg_words if w.strip() and a < duration]
        words += found or [Word(round(start, 3), round(min(end, duration), 3), " " + text)]
    return words


def cues_from_segments(segments, duration: float, line_chars: int = 16, max_lines: int = 2) -> list[Cue]:
    return split_words(segment_words(segments, duration), line_chars, max_lines)


def translate_span(model, audio: np.ndarray, start: float, end: float, language: str | None, hint: str | None = None) -> str:
    """audio의 [start, end] 구간만 영어로 (두 줄 자막의 둘째 줄).

    앞 줄 번역을 이어 주는 말(initial_prompt)로 줬더니 그 말을 되풀이했다 ("How is it? Isn't it pretty?" — 실측) → 쓰지 않는다.
    hint = 용어집에서 이 줄에 나오는 낱말의 영어 (곶감 → dried persimmon).
    """
    clip = audio[max(0, int((start - TRANSLATE_PAD) * SAMPLE_RATE)) : int((end + TRANSLATE_PAD) * SAMPLE_RATE)]
    if len(clip) < SAMPLE_RATE // 10:
        return ""
    segments, _info = model.transcribe(clip, task="translate", language=language, vad_filter=False, without_timestamps=True,
                                       condition_on_previous_text=False, hotwords=hint)
    return " ".join(s.text.strip() for s in segments).strip()


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

    def request(self, path: Path, task: str, audio_fn=None, layout: tuple[int, int] = (16, 2), terms=()) -> None:
        """path의 소리를 알아듣는다. audio_fn을 주면 그 함수가 돌려준 소리(48kHz 모노 float32)로 (컷편집 타임라인).

        layout = (한 줄 글자 수, 줄 수) — 자동 줄 나누기. terms = 용어집 (받아쓰기·번역 힌트).
        """
        self.busy_with = path
        self._jobs.put((path, task, audio_fn, layout, tuple(terms)))

    def _load(self, path: Path):
        if self._model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows 심볼릭 링크 경고 (동작엔 상관없음)
            from faster_whisper import WhisperModel

            done = speech_model_ready()
            self.status.emit(path, "음성 인식 모델 여는 중…" if done else "음성 인식 모델 내려받는 중 (처음 한 번, 약 3GB)…")
            try:
                self._model = WhisperModel(MODEL, device="cuda", compute_type="float16", download_root=str(MODEL_DIR))
            except Exception as e:  # NVIDIA GPU가 없거나 CUDA를 못 쓰면
                log.info("음성 인식 GPU를 못 써서 CPU로: %s", e)
                self._model = WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=str(MODEL_DIR))
        return self._model

    def _worker(self) -> None:
        while True:
            path, task, audio_fn, layout, terms = self._jobs.get()
            try:
                model = self._load(path)
                self.status.emit(path, "소리 읽는 중…")
                audio = load_audio(path) if audio_fn is None else resample_48k(audio_fn())
                duration = len(audio) / SAMPLE_RATE
                if duration == 0:
                    raise RuntimeError("소리가 없는 영상입니다")
                self.status.emit(path, "말 알아듣는 중…")
                bilingual = task == "bilingual"
                share = 0.5 if bilingual else 1.0  # 두 줄 자막은 번역이 나머지 절반
                english = task == "translate"
                hint = (", ".join(t.english for t in terms if t.english) or None) if english else speech_hint(terms)
                segments, info = model.transcribe(audio, task="translate" if english else "transcribe", hotwords=hint,
                                                  vad_filter=True, condition_on_previous_text=False, word_timestamps=True)
                found = []
                for segment in segments:  # 하나씩 나오므로 진행률을 보여 줄 수 있다
                    found.append((segment.start, segment.end, segment.text,
                                  [(w.start, w.end, w.word) for w in (segment.words or [])]))
                    self.progress.emit(path, share * min(1.0, segment.end / duration))
                cues = cues_from_segments(found, duration, *layout)
                if bilingual and info.language != "en":  # 영어로 말했으면 번역할 것이 없다
                    for i, cue in enumerate(cues):
                        self.status.emit(path, f"영어로 옮기는 중… {i + 1}/{len(cues)}")
                        translation = translate_span(model, audio, cue.start, cue.end, info.language, translation_hint(terms, cue.text))
                        cues[i] = Cue(cue.start, cue.end, cue.text, translation, cue.words)
                        self.progress.emit(path, share + (1 - share) * (i + 1) / len(cues))
                self.busy_with = None
                self.done.emit(path, cues)
            except Exception as e:
                log.exception("음성 인식 실패: %s", path)
                self.busy_with = None
                self.failed.emit(path, str(e))
