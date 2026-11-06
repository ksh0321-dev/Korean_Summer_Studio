"""AI 탭 화면용 그림(샘플 컷 3장 · 장면 미리보기)을 작업 스레드에서 만든다.

예전엔 UI 스레드에서 화면 전체(4K)에 효과를 입혀 슬라이더를 한 번 움직일 때마다 약 0.9초 멈췄다 (실측: 4K, 효과 5개, 샘플 4장).
- 보여 줄 크기로 먼저 줄이고 효과를 입힌다: 효과는 얼굴 크기(눈 사이 거리) 기준이라 줄여도 모습이 같다
- 작업 스레드 하나에서, 밀린 요청은 버리고 가장 최근 것만 (슬라이더를 끄는 동안 쌓이지 않게)
"""

import cv2
import numpy as np

from ks_studio.core.beauty import BeautySettings
from ks_studio.services.beautify import apply, regions

WORK_SCALE = 2  # 샘플 컷은 보여 줄 크기의 2배 해상도에서 효과를 계산한다 (줄였을 때 또렷하게)
GAP = 6


def side_by_side(*images: np.ndarray, gap: int = GAP) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    out = np.full((h, sum(i.shape[1] for i in images) + gap * (len(images) - 1), 3), 24, np.uint8)
    x = 0
    for image in images:
        out[: image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return out


def face_square(shape, landmarks: np.ndarray) -> tuple[int, int, int, int]:
    """얼굴 둘레 정사각형 (left, top, right, bottom). 효과(턱선 당기기 등)가 닿는 자리까지 들어가게 넉넉히."""
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8
    h, w = shape[:2]
    return int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))


def face_crop(rgb: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
    left, top, right, bottom = face_square(rgb.shape, landmarks)
    return cv2.resize(rgb[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA)


def sample_strip(rgb: np.ndarray, landmarks: np.ndarray, settings: BeautySettings, size: int) -> np.ndarray:
    """샘플 컷 한 줄: 원본 · 적용 범위 · 적용 후 (각 size×size). 얼굴만 잘라 줄인 뒤 효과를 입힌다."""
    left, top, right, bottom = face_square(rgb.shape, landmarks)
    crop = rgb[top:bottom, left:right]
    scale = min(1.0, size * WORK_SCALE / max(1, crop.shape[1]))
    small = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    local = (landmarks - (left, top)) * scale
    frame_width = rgb.shape[1] * scale  # 뒤쪽 작은 얼굴 판정(화면 폭 대비 얼굴 크기)이 원래와 같게
    images = (small, regions(small, local, settings), apply(small, local, settings, frame_width=frame_width))
    return side_by_side(*(cv2.resize(i, (size, size), interpolation=cv2.INTER_AREA) for i in images))


def preview_pair(rgb: np.ndarray, landmarks: np.ndarray | None, settings: BeautySettings, box: tuple[int, int],
                 show_points: bool) -> np.ndarray:
    """장면 미리보기: 보정 전 | 보정 후를 box(가로, 세로) 안에 들어가게. 줄인 그림에 효과를 입힌다."""
    half_w = max(1, (box[0] - GAP) // 2)
    scale = min(half_w / rgb.shape[1], max(1, box[1]) / rgb.shape[0], 1.0)
    size = (max(2, round(rgb.shape[1] * scale)), max(2, round(rgb.shape[0] * scale)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    local = None if landmarks is None else landmarks * scale
    after = apply(small, local, settings, frame_width=rgb.shape[1] * scale)
    if show_points and local is not None:
        after = after.copy()
        for x, y in local.astype(int):
            cv2.circle(after, (int(x), int(y)), 2, (0, 255, 0), -1)
    return side_by_side(small, after)
