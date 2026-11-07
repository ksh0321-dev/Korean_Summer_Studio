"""운영체제·하드웨어에 따라 고르는 것들을 한 곳에: AI 모델 실행 장치, 영상 인코더 후보, 폴더 열기.

- Windows·Linux + NVIDIA: CUDA (onnxruntime-gpu) · NVENC
- macOS (Apple Silicon): Core ML (onnxruntime 기본 휠에 들어 있다) · VideoToolbox
- 어디서나 마지막은 CPU (느리지만 돈다)
되는 것을 차례로 열어 보고 안 되면 다음으로 — 그래서 한 코드로 세 OS에서 돈다.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MAC = sys.platform == "darwin"
SOFTWARE_ENCODING = False  # 설정 '성능' 탭: True면 녹화를 뺀 모든 인코딩을 CPU 인코더로만 (set_software_encoding)
WINDOWS = sys.platform == "win32"
_preloaded = False


def onnx_providers(coreml: bool = True) -> list[str]:
    """onnxruntime 세션에 줄 실행 장치 (좋은 것부터). coreml=False: 입력 크기가 바뀌는 모델(MI-GAN 등)은 CPU로."""
    global _preloaded
    import onnxruntime

    if not _preloaded and hasattr(onnxruntime, "preload_dlls"):
        _preloaded = True
        try:
            onnxruntime.preload_dlls()  # pip로 깐 CUDA·cuDNN DLL을 먼저 찾아 둔다 (없으면 CUDA 세션이 안 열렸다 — 실측)
        except Exception as e:  # CUDA판이 아닌 onnxruntime(맥 등)
            log.debug("preload_dlls 건너뜀: %s", e)
    available = onnxruntime.get_available_providers()
    wanted = ["CUDAExecutionProvider", *(["CoreMLExecutionProvider"] if coreml else []), "CPUExecutionProvider"]
    return [p for p in wanted if p in available] or ["CPUExecutionProvider"]


def onnx_session(path: Path, coreml: bool = True):
    """모델 세션: 좋은 장치로 열어 보고, 그 장치가 이 모델을 못 열면 CPU로."""
    import onnxruntime

    providers = onnx_providers(coreml)
    try:
        return onnxruntime.InferenceSession(str(path), providers=providers)
    except Exception as e:
        if providers == ["CPUExecutionProvider"]:
            raise
        log.warning("%s를 %s로 못 열어 CPU로: %s", Path(path).name, providers[0], e)
        return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def device_name(providers: list[str]) -> str:
    """rtmlib에 줄 장치 이름."""
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    if "CoreMLExecutionProvider" in providers:
        return "mps"  # rtmlib: mps → CoreMLExecutionProvider
    return "cpu"


def hardware_encoders(codec: str, bitrate: str) -> tuple:
    """GPU 인코더 후보 (codec: "h264"·"hevc"). NVENC 설정은 각자 들고 있고, 여기는 맥 VideoToolbox만 — 비트레이트로."""
    return ((f"{codec}_videotoolbox", {"b": bitrate, "allow_sw": "1"}),)


def open_folder(path: Path, select: bool = False) -> None:
    """파일 탐색기(Windows)·Finder(macOS)·파일 관리자(Linux)로 연다. select면 그 파일을 골라 둔 채로."""
    path = Path(path).resolve()
    if WINDOWS:
        if select and path.exists():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(path if path.is_dir() else path.parent)  # noqa: S606
    elif MAC:
        subprocess.Popen(["open", "-R", str(path)] if select and path.exists() else ["open", str(path if path.is_dir() else path.parent)])
    else:
        subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])


def set_software_encoding(on: bool) -> None:
    global SOFTWARE_ENCODING
    SOFTWARE_ENCODING = on
