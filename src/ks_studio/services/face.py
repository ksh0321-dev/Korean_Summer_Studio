"""얼굴 찾기·106점 랜드마크·얼굴 인식값 (InsightFace + onnxruntime GPU). AI 보정 작업 스레드 하나에서만 쓴다.

- 모델은 처음 쓸 때 프로젝트 models/에 내려받는다 (약 600MB, 저장소에 올리지 않음)
- 보정할 사람(주인공)은 인식값으로 정한다. 정하지 않았으면 가장 큰 얼굴
- 실측 (RTX 5080, CUDA): 1080p 한 프레임 얼굴 찾기 + 랜드마크 약 12ms (인식값까지 하면 얼굴마다 조금 더)
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from ks_studio.app.paths import MODELS_DIR
from ks_studio.core.beauty import frontal_weight, size_weight
from ks_studio.core.people import cluster, match

log = logging.getLogger(__name__)

MODEL_PACK = "buffalo_l"
DETECT_SIZE = (640, 640)
SAMPLE_PROBES = 24  # 샘플 컷 후보로 훑어볼 장면 수
MIN_DETECTION_SCORE = 0.6  # 실측: 뒤통수를 얼굴로 착각한 것 0.53, 진짜 얼굴 0.66~0.86
PERSON_FRONTAL = 0.5  # '보정할 사람' 목록은 이만큼 정면인 얼굴로만 만든다 (귀·옆머리만 보이는 장면이 따로 묶이지 않게)
LANDMARKS = 106


def ai_unavailable_reason() -> str | None:
    """AI 패키지가 없으면 설치 안내, 있으면 None."""
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        return f"AI 보정 패키지가 없습니다 ({e.name}). 설치: pip install -e .[ai]"
    return None


@dataclass(frozen=True)
class FoundFace:
    landmarks: np.ndarray  # (106, 2)
    embedding: np.ndarray  # (512,) 길이 1

    @property
    def area(self) -> float:
        return float(np.ptp(self.landmarks[:, 0]) * np.ptp(self.landmarks[:, 1]))


def choose(faces: list[FoundFace], target: np.ndarray | None) -> FoundFace | None:
    """보정할 얼굴: 주인공을 정했으면 그 사람 (그 장면에 없으면 None), 아니면 가장 큰 얼굴."""
    if not faces:
        return None
    if target is None:
        return max(faces, key=lambda f: f.area)
    index = match([f.embedding for f in faces], target)
    return faces[index] if index is not None else None


def _estimate_norm(lmk: np.ndarray, image_size: int = 112, mode: str = "arcface") -> np.ndarray:
    """insightface.utils.face_align.estimate_norm과 같은 계산을 scikit-image 새 API(from_estimate)로.

    insightface 2.1은 0.26부터 폐기 예정인 SimilarityTransform.estimate를 써서 경고가 난다 (2.2에서 없어짐).
    라이브러리 파일은 다시 깔면 되돌아가므로 고치지 않고, 모델을 열 때 이 함수로 바꿔 끼운다.
    """
    from insightface.utils.face_align import arcface_dst
    from skimage.transform import SimilarityTransform

    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio, diff_x = image_size / 112.0, 0.0
    else:
        ratio = image_size / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = SimilarityTransform.from_estimate(lmk, dst)
    if not tform:  # 점이 한 곳에 몰리는 등 계산 불가 — 실제 얼굴에선 생기지 않는다
        return np.eye(3)[0:2]
    return tform.params[0:2, :]


class FaceDetector:
    """처음 부를 때 모델을 연다 (내려받기 포함이면 수십 초)."""

    def __init__(self) -> None:
        self._app = None

    def _load(self):
        if self._app is None:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align

            face_align.estimate_norm = _estimate_norm  # 폐기 예정 API 경고 없이 (위 설명)

            app = FaceAnalysis(
                name=MODEL_PACK,
                root=str(MODELS_DIR),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                allowed_modules=["detection", "landmark_2d_106", "recognition"],
            )
            app.prepare(ctx_id=0, det_size=DETECT_SIZE)
            self._app = app
        return self._app

    def faces(self, rgb: np.ndarray) -> list[FoundFace]:
        found = self._load().get(np.ascontiguousarray(rgb[:, :, ::-1]))  # InsightFace는 BGR
        return [
            FoundFace(f.landmark_2d_106.astype(np.float64), f.normed_embedding.astype(np.float32))
            for f in found
            if f.det_score >= MIN_DETECTION_SCORE
        ]


@dataclass(frozen=True)
class Probe:
    """샘플 후보 장면 하나와 그 장면의 얼굴들."""

    frame: int
    rgb: np.ndarray
    faces: tuple[FoundFace, ...]


@dataclass(frozen=True)
class Person:
    """후보 장면들에 나온 한 사람."""

    embedding: np.ndarray  # 대표 인식값 (가장 정면·큰 얼굴의 것)
    appearances: int
    probe: int  # 대표 얼굴이 있는 후보 장면 번호
    face: FoundFace


@dataclass(frozen=True)
class Sample:
    """샘플 컷 하나: 영상의 frame번째 장면과 보정할 얼굴."""

    frame: int
    rgb: np.ndarray
    landmarks: np.ndarray


def quality(face: FoundFace, width: int) -> float:
    """보여줄 만한 정도: 클수록, 정면일수록 높다 (옆얼굴·뒤쪽 작은 얼굴은 0)."""
    return float(np.ptp(face.landmarks[:, 0])) * frontal_weight(face.landmarks) * size_weight(face.landmarks, width)


def frame_at(path: Path, frame: int) -> np.ndarray | None:
    """frame번째 프레임(0부터) RGB. 고정 fps 파일(녹화본·편집용 사본·보정본) 기준."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        container.seek(int(frame / fps / stream.time_base), stream=stream, backward=True)
        for decoded in container.decode(stream):
            if round(decoded.pts * stream.time_base * fps) >= frame:
                return decoded.to_ndarray(format="rgb24")
    return None


def keyframe_near(path: Path, frame: int) -> tuple[int, np.ndarray] | None:
    """frame 바로 앞의 키프레임 (번호, RGB). 키프레임만 풀어서 빠르다.

    실측: 정확한 프레임까지 풀면 1080p 장면당 0.5초, 4K는 2.7초 (앞 키프레임부터 수십 장을 풀어야 한다).
    샘플 후보는 정확한 프레임일 필요가 없다.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        fps = float(stream.average_rate)
        container.seek(int(frame / fps / stream.time_base), stream=stream, backward=True)
        for decoded in container.decode(stream):
            return round(decoded.pts * stream.time_base * fps), decoded.to_ndarray(format="rgb24")
    return None


def scan(detector: FaceDetector, path: Path, on_progress=None) -> list[Probe]:
    """영상 전체에서 고르게 SAMPLE_PROBES 장면을 훑어 얼굴을 찾는다 (샘플 컷·사람 목록의 재료).

    on_progress(끝낸 수, 전체 수)는 장면마다 불린다.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or round(float(container.duration / 1_000_000) * float(stream.average_rate))
    targets = sorted({round(total * (i + 0.5) / SAMPLE_PROBES) for i in range(SAMPLE_PROBES)})
    slot = total / SAMPLE_PROBES
    probes, seen = [], set()
    for done, target in enumerate(targets, 1):
        found = keyframe_near(path, target)
        # 키프레임 간격이 길면(몇 초) 여러 후보가 같은 장면이 된다 → 그 후보만 정확한 프레임까지 푼다
        if found is None or found[0] in seen or target - found[0] > slot / 2:
            rgb = frame_at(path, target)
            found = (target, rgb) if rgb is not None else None
        if found is not None and found[0] not in seen:
            seen.add(found[0])
            probes.append(Probe(found[0], found[1], tuple(detector.faces(found[1]))))
        if on_progress:
            on_progress(done, len(targets))
    return probes


SAMPLES_VERSION = 1  # 찾기 방식(MIN_DETECTION_SCORE 등)을 바꾸면 올려서 예전 캐시를 버린다
JPEG_QUALITY = 92


def samples_cache(cache_dir: Path, media_name: str) -> Path:
    """샘플 후보 장면 캐시 (<프로젝트>/cache/ai_samples/). 같은 영상을 다시 고르면 얼굴 찾기 없이 바로 보여 준다."""
    return cache_dir / "ai_samples" / f"{media_name}.v{SAMPLES_VERSION}.npz"


def save_probes(path: Path, probes: list[Probe]) -> None:
    """후보 장면(JPEG로 줄여서)과 얼굴(랜드마크·인식값)을 한 파일로. 1080p 24장 ≈ 수 MB."""
    import cv2

    encoded = [cv2.imencode(".jpg", p.rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].reshape(-1) for p in probes]
    faces = [f for p in probes for f in p.faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    np.savez(
        tmp,
        frames=np.array([p.frame for p in probes], np.int64),
        jpeg=np.concatenate(encoded) if encoded else np.zeros(0, np.uint8),
        jpeg_sizes=np.array([len(e) for e in encoded], np.int64),
        face_counts=np.array([len(p.faces) for p in probes], np.int64),
        landmarks=np.array([f.landmarks for f in faces], np.float64).reshape(-1, LANDMARKS, 2),
        embeddings=np.array([f.embedding for f in faces], np.float32).reshape(len(faces), -1),
    )
    tmp.replace(path)


def load_probes(path: Path) -> list[Probe] | None:
    """캐시가 없거나 깨졌으면 None (그러면 다시 찾는다)."""
    import cv2

    try:
        data = np.load(path)
        frames, sizes, counts = data["frames"], data["jpeg_sizes"], data["face_counts"]
        jpeg, landmarks, embeddings = data["jpeg"], data["landmarks"], data["embeddings"]
    except (OSError, KeyError, ValueError):
        return None
    probes, offset, face = [], 0, 0
    for frame, size, count in zip(frames, sizes, counts):
        bgr = cv2.imdecode(jpeg[offset : offset + size], cv2.IMREAD_COLOR)
        offset += size
        if bgr is None:
            return None
        found = tuple(FoundFace(landmarks[face + k], embeddings[face + k]) for k in range(count))
        face += count
        probes.append(Probe(int(frame), np.ascontiguousarray(bgr[:, :, ::-1]), found))
    return probes


def people(probes: list[Probe]) -> list[Person]:
    """후보 장면에 나온 사람들, 많이 나온 순. 정면에 가까운 얼굴로만 묶는다."""
    entries = [
        (i, face)
        for i, probe in enumerate(probes)
        for face in probe.faces
        if frontal_weight(face.landmarks) >= PERSON_FRONTAL and size_weight(face.landmarks, probe.rgb.shape[1]) > 0
    ]
    labels = cluster([face.embedding for _, face in entries])
    result = []
    for label in set(labels):
        members = [entries[k] for k, lab in enumerate(labels) if lab == label]
        best_probe, best_face = max(members, key=lambda m: quality(m[1], probes[m[0]].rgb.shape[1]))
        result.append(Person(best_face.embedding, len(members), best_probe, best_face))
    return sorted(result, key=lambda p: (-p.appearances, -p.face.area))


def pick_samples(probes: list[Probe], target: np.ndarray | None, count: int = 4) -> list[Sample]:
    """보정할 사람이 잘 보이는 장면 count개. 영상을 count 구간으로 나눠 구간마다 하나씩 (여러 곳의 모습을 보게)."""
    found = []
    for probe in probes:
        face = choose(list(probe.faces), target)
        if face is not None and quality(face, probe.rgb.shape[1]) > 0:
            found.append((quality(face, probe.rgb.shape[1]), Sample(probe.frame, probe.rgb, face.landmarks)))
    if not found:
        return []
    last = max(p.frame for p in probes) + 1
    bins: list[list] = [[] for _ in range(count)]
    for entry in found:
        bins[min(count - 1, entry[1].frame * count // last)].append(entry)
    chosen = [max(b, key=lambda e: e[0]) for b in bins if b]
    rest = sorted((e for e in found if e not in chosen), key=lambda e: -e[0])
    return sorted((s for _, s in chosen + rest[: count - len(chosen)]), key=lambda s: s.frame)


def frames_from(path: Path, start: int = 0):
    """고정 fps 파일의 start번째 프레임부터 끝까지 (번호, 프레임). 필요한 만큼 받고 그만 받으면 된다."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start <= 0:  # 처음부터면 차례로 센다
            yield from enumerate(container.decode(stream))
            return
        fps = float(stream.average_rate)
        container.seek(int(start / fps / stream.time_base), stream=stream, backward=True)
        for decoded in container.decode(stream):  # 앞 키프레임부터 풀리므로 start 전은 건너뛴다
            index = round(decoded.pts * stream.time_base * fps)
            if index >= start:
                yield index, decoded


def analyze_video(
    detector: FaceDetector, path: Path, target: np.ndarray | None, on_progress=None, start: int = 0, count: int | None = None
) -> np.ndarray:
    """start번째부터 count장(None = 끝까지) 프레임에서 보정할 얼굴의 랜드마크 (장 수, 106, 2). 그 사람이 없는 프레임은 NaN."""
    rows = []
    with av.open(str(path)) as container:
        total = count or container.streams.video[0].frames or 0
    for i, (_index, decoded) in enumerate(frames_from(path, start)):
        if count is not None and i >= count:
            break
        face = choose(detector.faces(decoded.to_ndarray(format="rgb24")), target)
        rows.append(face.landmarks if face is not None else np.full((LANDMARKS, 2), np.nan))
        if on_progress and total and i % 15 == 0:
            on_progress(min(1.0, i / total))
    return np.stack(rows) if rows else np.zeros((0, LANDMARKS, 2))


def track_file(cache_dir: Path, media_name: str, target: np.ndarray | None) -> Path:
    """영상 전체 분석 결과 파일. 보정할 사람마다 따로 (사람을 바꾸면 다시 분석)."""
    who = "largest" if target is None else hashlib.sha1(np.round(target, 3).astype(np.float32).tobytes()).hexdigest()[:10]
    return cache_dir / f"{media_name}.faces_{who}.npy"
