"""`KS Studio.exe --selftest`: 이 PC에서 필요한 것이 다 도는지 확인하고 user/logs/selftest.txt에 적는다.

다른 PC(Python·CUDA 없음)에 배포했을 때 GPU가 실제로 쓰이는지 한 번에 본다:
글꼴 · Qt 영상 재생 · 영상 인코더(NVENC/CPU) · 얼굴 모델(onnxruntime CUDA) · 몸매 모델(rtmlib) · 배경 메꾸기(MI-GAN·LaMa)
· 음성 인식(ctranslate2 CUDA).
모델 파일이 아직 없으면 그 항목은 '모델 없음'으로 건너뛴다 (내려받지 않는다).
"""

import platform
import time
import traceback

from ks_studio import __version__
from ks_studio.app.paths import APP_DIR, FROZEN, MODELS_DIR, USER_DIR


def _check(name: str, fn, lines: list[str]) -> None:
    started = time.monotonic()
    try:
        detail = fn()
        lines.append(f"OK    {name}: {detail} ({time.monotonic() - started:.1f}s)")
    except Exception as e:  # 무엇이 안 되는지 적는 것이 목적
        lines.append(f"FAIL  {name}: {type(e).__name__}: {e}")
        lines.append("      " + traceback.format_exc().strip().replace("\n", "\n      "))


def _fonts():
    from ks_studio.app.fonts import family, register_fonts

    return f"{family()} (등록 {register_fonts()})"


def _player():
    from PySide6.QtMultimedia import QMediaPlayer

    QMediaPlayer()
    return "QMediaPlayer 만들기"


def _encoder():
    from ks_studio.services.optimize import video_encoder

    return video_encoder((1920, 1080), 30)[0]


def _faces():
    import numpy as np
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if not (MODELS_DIR / "models" / "buffalo_l").exists():
        return f"providers {providers} · 모델 없음 (처음 쓸 때 내려받음)"
    from ks_studio.services.face import FaceDetector

    detector = FaceDetector()
    detector.faces(np.zeros((480, 640, 3), np.uint8))
    used = detector._load().models["detection"].session.get_providers()
    return f"사용 {used[0]} · 가능 {providers}"


def _body():
    import numpy as np
    import rtmlib  # noqa: F401 — exe에 들어갔는지 (모델이 없어도 확인)

    from ks_studio.services.body import BodyAnalyzer, body_models_ready

    if not body_models_ready():
        return "rtmlib 있음 · 모델 없음 (AI 탭 '모델'에서 받음)"
    analyzer = BodyAnalyzer.shared()
    rgb = np.zeros((960, 540, 3), np.uint8)
    analyzer.parts(rgb)
    analyzer.keypoints(rgb, (0, 0, 540, 960))
    return f"사용 {analyzer._parts.get_providers()[0]}"


def _inpaint():
    import numpy as np

    from ks_studio.services.inpaint import FILL_MODELS, INPAINT_DIR, _run_model, _session, fill_model_ready

    found = []
    for kind in FILL_MODELS:
        if not fill_model_ready(kind):
            found.append(f"{kind} 모델 없음")
            continue
        _run_model(kind, INPAINT_DIR, [np.zeros((512, 512, 3), np.uint8)], [np.ones((512, 512), np.uint8)])
        found.append(f"{kind} {_session(kind, INPAINT_DIR).get_providers()[0]}")
    return " · ".join(found)


def _speech():
    import numpy as np

    from ks_studio.services.speech import speech_model_ready, use_mlx

    if use_mlx():  # 맥: mlx-whisper (Apple Silicon GPU)
        if not speech_model_ready():
            return "mlx-whisper · 모델 없음 (처음 쓸 때 내려받음)"
        from ks_studio.services.speech import MODEL_DIR, MlxWhisper

        MlxWhisper(MODEL_DIR).transcribe(np.zeros(16000, np.float32))
        return "mlx-whisper (Apple Silicon GPU)"
    import ctranslate2

    devices = ctranslate2.get_cuda_device_count()
    from ks_studio.services.speech import MODEL, MODEL_DIR

    if not (MODEL_DIR / f"models--Systran--faster-whisper-{MODEL}").exists():
        return f"CUDA 장치 {devices}개 · 모델 없음 (처음 쓸 때 내려받음)"
    from faster_whisper import WhisperModel

    device = "cuda" if devices else "cpu"
    model = WhisperModel(MODEL, device=device, compute_type="float16" if devices else "int8", download_root=str(MODEL_DIR))
    segments, _ = model.transcribe(np.zeros(16000, np.float32), vad_filter=True)
    list(segments)
    return f"{device} (CUDA 장치 {devices}개)"


def run() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841 (글꼴·영상 재생에 필요)
    lines = [f"KS Studio {__version__} 자체 점검 · {'exe' if FROZEN else '소스'} · {platform.platform()}", f"앱 폴더: {APP_DIR}"]
    for name, fn in (("글꼴", _fonts), ("영상 재생", _player), ("영상 인코더", _encoder), ("얼굴 모델", _faces), ("몸매 모델", _body),
                     ("배경 메꾸기", _inpaint), ("음성 인식", _speech)):
        _check(name, fn, lines)
    out = USER_DIR / "logs" / "selftest.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not any(line.startswith("FAIL") for line in lines) else 1
