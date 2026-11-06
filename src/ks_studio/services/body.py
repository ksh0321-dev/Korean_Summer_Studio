"""몸매 보정·얼굴 작게에 쓰는 사람 분석: 사람 분리(배경·머리카락·몸 피부·얼굴 피부·옷·기타) + 몸의 점(RTMPose 17점).

- 사람 분리: MediaPipe selfie multiclass 256×256 (Apache 2.0, ONNX 변환본) — 사람 층만 변형하고 배경은 그대로 두려고
- 몸의 점: RTMPose-m body7 256×192 (Apache 2.0, OpenMMLab) — rtmlib로 앞뒤 처리. 사람 상자는 사람 분리에서 얻는다
- 모델은 처음 쓸 때 models/body/에 내려받는다 (약 70MB, 저장소에 올리지 않음)
- 실측 (RTX 5080, 4K 세로 프레임): 사람 분리 약 50ms(대부분 4K를 줄이는 데), 몸의 점 약 15ms. 첫 호출만 0.5초 남짓
- 얼굴 모델·보정 렌더·미리보기 스레드가 같이 쓴다 → 세션은 하나만 열고(잠금) run은 여러 스레드에서 불러도 된다

COCO 17점: 0 코 · 1/2 눈 · 3/4 귀 · 5/6 어깨 · 7/8 팔꿈치 · 9/10 손목 · 11/12 엉덩이 · 13/14 무릎 · 15/16 발목 (왼/오른 = 사람 기준)
"""

import logging
import os
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from ks_studio.app.paths import MODELS_DIR

log = logging.getLogger(__name__)

BODY_DIR = MODELS_DIR / "body"
POSE_FILE, PARTS_FILE = "rtmpose-m_body7_256x192.onnx", "selfie_multiclass_256.onnx"
POSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
PARTS_URL = "https://huggingface.co/senty-au/selfie_multiclass_256x256-ONNX/resolve/main/onnx/model.onnx"
BODY_MODEL_MB = 70
PARTS_SIZE = 256
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)
KEYPOINT_MIN_SCORE = 0.3
CHUNK = 1 << 20


def body_models_ready(folder: Path = BODY_DIR) -> bool:
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in (POSE_FILE, PARTS_FILE))


def _fetch(url: str, target: Path, on_progress=None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "KS-Studio"})
    with urllib.request.urlopen(request, timeout=30) as response, open(target, "wb") as out:
        total, done = int(response.headers.get("Content-Length") or 0), 0
        while chunk := response.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if on_progress and total:
                on_progress(min(1.0, done / total))


def download_body_models(on_progress=None, folder: Path = BODY_DIR, pose_url: str = POSE_URL, parts_url: str = PARTS_URL) -> None:
    """두 모델을 받아 둔다. 받는 중엔 임시 이름이라 끊겨도 '받은 것'으로 보이지 않는다."""
    folder.mkdir(parents=True, exist_ok=True)
    report = (lambda base, share: (lambda r: on_progress(base + share * r))) if on_progress else (lambda base, share: None)
    archive = folder / (POSE_FILE + ".zip.part")
    try:
        _fetch(pose_url, archive, report(0.0, 0.75))  # RTMPose 묶음이 대부분 (약 50MB)
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".onnx"))
            (folder / (POSE_FILE + ".part")).write_bytes(zf.read(name))
        os.replace(folder / (POSE_FILE + ".part"), folder / POSE_FILE)
        _fetch(parts_url, folder / (PARTS_FILE + ".part"), report(0.75, 0.25))
        os.replace(folder / (PARTS_FILE + ".part"), folder / PARTS_FILE)
    finally:
        archive.unlink(missing_ok=True)
        for leftover in folder.glob("*.part"):
            leftover.unlink(missing_ok=True)


class BodyAnalyzer:
    """사람 분리 + 몸의 점. 처음 쓸 때 모델을 연다 (GPU가 없으면 CPU)."""

    _shared = None
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "BodyAnalyzer":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def __init__(self, folder: Path = BODY_DIR) -> None:
        self._folder = folder
        self._lock = threading.Lock()
        self._parts = None
        self._pose = None

    def _load(self):
        with self._lock:
            if self._parts is None:
                import onnxruntime
                from rtmlib import RTMPose

                if hasattr(onnxruntime, "preload_dlls"):
                    onnxruntime.preload_dlls()  # pip로 깐 CUDA·cuDNN DLL을 먼저 찾아 둔다 (없으면 CUDA 세션이 안 열렸다 — 실측)
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._parts = onnxruntime.InferenceSession(str(self._folder / PARTS_FILE), providers=providers)
                cuda = "CUDAExecutionProvider" in self._parts.get_providers()
                self._pose = RTMPose(str(self._folder / POSE_FILE), model_input_size=(192, 256), backend="onnxruntime",
                                     device="cuda" if cuda else "cpu")
        return self._parts, self._pose

    def parts(self, rgb: np.ndarray) -> np.ndarray:
        """(256, 256, 6) 부위 확률 (0~1, float16). 원래 크기로는 쓰는 쪽에서 늘린다 (늘린 확률이 곧 부드러운 경계)."""
        import cv2

        session, _ = self._load()
        small = cv2.resize(rgb, (PARTS_SIZE, PARTS_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255
        logits = session.run(None, {session.get_inputs()[0].name: small})[0][0]
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs.astype(np.float16)

    def keypoints(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """(17, 3) 몸의 점 (x, y, 점수). box = 사람 상자 (x0, y0, x1, y1)."""
        _, pose = self._load()
        points, scores = pose(np.ascontiguousarray(rgb[:, :, ::-1]), bboxes=[list(box)])
        return np.concatenate([points[0], scores[0][:, None]], axis=1).astype(np.float64)


def person_box(parts: np.ndarray, width: int, height: int, near: tuple[float, float] | None = None) -> tuple[int, int, int, int] | None:
    """사람 영역 상자 (원래 크기 좌표). near(얼굴 자리)가 있으면 그 사람 덩어리만."""
    import cv2

    person = (parts[..., BACKGROUND] < 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(person, connectivity=8)
    if count <= 1:
        return None
    if near is not None:
        nx, ny = int(near[0] / width * PARTS_SIZE), int(near[1] / height * PARTS_SIZE)
        label = labels[min(PARTS_SIZE - 1, max(0, ny)), min(PARTS_SIZE - 1, max(0, nx))]
    else:
        label = 0
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[label, :4]
    sx, sy = width / PARTS_SIZE, height / PARTS_SIZE
    return int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)


def layer(parts: np.ndarray, width: int, height: int, classes: tuple[int, ...]) -> np.ndarray:
    """그 부위들의 확률을 원래 크기로 (float32 0~1)."""
    import cv2

    prob = parts[..., list(classes)].astype(np.float32).sum(axis=-1)
    return np.clip(cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR), 0, 1)
