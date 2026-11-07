"""얼굴 흐리기를 화소에 입힌다: 모자이크 또는 흐림, 얼굴 둘레 타원 (가장자리는 부드럽게).

영상 프레임(YUV)은 판(Y·U·V)마다 그 자리만 흐린다 — 화면 전체를 RGB로 바꿨다 되돌리지 않는다 (빠르고 나머지 화소는 그대로).
미리보기(RGB)도 같은 함수로 흐려서 보이는 모습이 같다.
"""

import av
import cv2
import numpy as np

from ks_studio.services.yuv import from_planes, planes

MOSAIC_BLOCKS = 9  # 모자이크: 얼굴 폭에 칸 9개 (알아볼 수 없을 만큼, 너무 거칠지 않게)
BLUR_SIGMA = 1 / 6  # 흐림: 얼굴 폭의 1/6
FEATHER = 0.12  # 타원 가장자리를 상자 크기의 12%만큼 부드럽게


def _ellipse(h: int, w: int) -> np.ndarray:
    """상자에 꽉 찬 타원 (안 1 → 밖 0, 가장자리 부드럽게)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xs + 0.5 - w / 2) / (w / 2)) ** 2 + ((ys + 0.5 - h / 2) / (h / 2)) ** 2
    return np.clip((1 - d) / (2 * FEATHER), 0, 1)


def _hidden(patch: np.ndarray, face_w: float, style: str) -> np.ndarray:
    h, w = patch.shape[:2]
    if style == "blur":
        sigma = max(1.0, face_w * BLUR_SIGMA)
        return cv2.GaussianBlur(patch, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    block = max(2.0, face_w / MOSAIC_BLOCKS)
    small = cv2.resize(patch, (max(1, round(w / block)), max(1, round(h / block))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_image(image: np.ndarray, boxes, style: str = "mosaic", scale: float = 1.0) -> None:
    """image(판 하나 또는 RGB)의 상자들을 제자리에서 흐린다. scale = 상자 좌표 → 이 판 좌표 (U·V 판은 0.5)."""
    h, w = image.shape[:2]
    for box in boxes:
        x0, y0 = max(0, int(box[0] * scale)), max(0, int(box[1] * scale))
        x1, y1 = min(w, int(np.ceil(box[2] * scale))), min(h, int(np.ceil(box[3] * scale)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = image[y0:y1, x0:x1]
        # 상자가 화면 끝에 걸려 잘렸어도 타원은 원래 상자 모양으로
        full_w, full_h = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
        mask = _ellipse(max(2, round(full_h)), max(2, round(full_w)))
        oy, ox = y0 - int(box[1] * scale), x0 - int(box[0] * scale)
        mask = mask[max(0, oy) : max(0, oy) + (y1 - y0), max(0, ox) : max(0, ox) + (x1 - x0)]
        if mask.shape != region.shape[:2]:
            mask = cv2.resize(mask, (x1 - x0, y1 - y0))
        hidden = _hidden(region, full_w * 0.75, style)  # 상자는 얼굴보다 크게 잡혀 있다 (core.face_blur.GROW)
        alpha = mask[..., None] if region.ndim == 3 else mask
        region[:] = np.clip(region + (hidden.astype(np.float32) - region) * alpha + 0.5, 0, 255).astype(np.uint8)


def blur_frame(frame: av.VideoFrame, boxes, style: str = "mosaic") -> av.VideoFrame:
    """영상 프레임의 얼굴들을 흐린 새 프레임 (상자가 없으면 그대로)."""
    if not boxes:
        return frame
    if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    y, u, v = planes(frame)
    blur_image(y, boxes, style)
    blur_image(u, boxes, style, 0.5)
    blur_image(v, boxes, style, 0.5)
    return from_planes(y, u, v, frame)
