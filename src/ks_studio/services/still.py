"""사진 ↔ 영상: 사진을 가져오면 짧은 영상으로 바꿔 타임라인에 넣고, 지금 화면을 사진(썸네일)으로 저장한다.

사진을 영상으로 바꿔 두면 나머지(타임라인·자르기·AI 보정·내보내기)는 영상과 똑같이 다룰 수 있다.
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ks_studio.services.yuv import from_rgb, to_rgb

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PHOTO_SECONDS = 5.0  # 사진 한 장의 기본 길이 (타임라인에서 줄이고 늘인다 — 늘이려면 처음부터 길게)
PHOTO_FPS = 30
PHOTO_LONG_SIDE = 3840  # 이보다 큰 사진은 줄인다 (4K)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def read_image(path: Path) -> np.ndarray | None:
    """사진 → RGB 배열 (휴대폰 사진의 회전 정보까지 반영). 못 읽으면 None."""
    from PySide6.QtGui import QImage, QImageReader

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)  # EXIF 회전
    image = reader.read()
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    h, w = image.height(), image.width()
    return np.frombuffer(image.constBits(), np.uint8).reshape(h, image.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def image_to_video(path: Path, output: Path, seconds: float = PHOTO_SECONDS, fps: int = PHOTO_FPS) -> Path:
    """사진 한 장을 seconds초짜리 무음 영상으로 (짝수 크기, 4K까지). 만든 파일을 돌려준다."""
    from ks_studio.services.optimize import add_video_stream

    rgb = read_image(path)
    if rgb is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path.name}")
    h, w = rgb.shape[:2]
    scale = min(1.0, PHOTO_LONG_SIDE / max(w, h))
    size = (max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2))
    frame = from_rgb(rgb, "yuv420p", *size)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    try:
        with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            stream = add_video_stream(dst, size, fps)
            for k in range(max(1, round(seconds * fps))):
                frame.pts, frame.time_base = k, Fraction(1, fps)
                dst.mux(stream.encode(frame))
            dst.mux(stream.encode(None))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(output)
    return output


def save_still(rgb: np.ndarray, output: Path) -> Path:
    """RGB 배열 → JPG·PNG (확장자대로)."""
    from PySide6.QtGui import QImage

    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).save(str(output), quality=95):
        raise OSError(f"저장하지 못했습니다: {output}")
    return output


def timeline_still(project, workspace, sources, frame: int, with_overlays: bool = True) -> np.ndarray:
    """타임라인 frame번째 장면을 결과물 크기 그대로 (글자·자막까지 구워서) — 썸네일용."""
    from ks_studio.services.captions import OverlayBurner
    from ks_studio.services.render import TimelineRenderer, canvas_size

    size = canvas_size(project, workspace, sources)
    renderer = TimelineRenderer(project, workspace, sources, size, "yuv420p")
    for t, image, _ in renderer.frames(max(0, frame), with_audio=False):
        if with_overlays and (project.subtitles.cues or project.texts):
            image.pts = t
            image = OverlayBurner(project.subtitles, project.texts, size, project.fps)(image)
        return to_rgb(image)
    raise ValueError("타임라인이 비어 있습니다")
