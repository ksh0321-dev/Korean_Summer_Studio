"""AI 보정을 화소에 입힌다 (core.beauty가 정한 자리에): 피부·메이크업(색) → 얼굴형(국소 변형).

- apply: 한 프레임 (샘플 컷·미리보기·렌더가 같이 쓴다). 얼굴 주변 영역만 계산해서 빠르다
- render_beautified: 파일 → 보정본 파일 (영상만 다시 인코딩, 소리는 그대로 복사)
- AiWorker: 얼굴 찾기·분석·렌더를 스레드 하나에서 차례로 (GPU 모델을 한 스레드만 쓰게)
"""

import logging
import math
import os
import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.core.beauty import (
    BLUSH_COLOR,
    EFFECTS,
    EYESHADOW_COLORS,
    SLIM_RADIUS,
    BeautySettings,
    Ellipse,
    Pull,
    Zoom,
    brow_polygons,
    cheek_ellipses,
    cheekbone_pulls,
    chin_pulls,
    contour_shapes,
    eye_distance,
    eye_polygons,
    eye_zooms,
    eyelid_polygons,
    frontal_weight,
    lip_polygons,
    nose_length_pulls,
    nose_pulls,
    nostril_polygon,
    size_weight,
    skin_polygon,
    slim_pulls,
    smile_pulls,
    smooth_track,
    under_eye_ellipses,
)
from ks_studio.services.face import (
    FaceDetector,
    analyze_video,
    choose,
    frame_at,
    frames_from,
    load_probes,
    save_probes,
    scan,
    track_file,
)
from ks_studio.services.foreground import yield_to_playback
from ks_studio.services.optimize import add_video_stream, make_optimized
from ks_studio.services.yuv import from_planes, planes
from ks_studio.services.sources import AUDIO_RATE, cache_name

log = logging.getLogger(__name__)

BLUSH_MAX_ALPHA = 0.35  # 강도 100%일 때 볼 가운데에서 색이 섞이는 비율
SMOOTH_MAX = 0.85  # 매끈하게 100%: 흐린 피부를 85% 섞는다 (결이 조금 남게)
SMOOTH_EYES_PX = 120  # 피부 흐리기는 눈 사이 거리가 이 정도가 되게 줄여서 한다 (4K 얼굴도 같은 모습·빠르게)
BRIGHTEN_MAX = 0.12  # 톤 밝게 100%: 남은 밝기 여유의 12%만큼
DARK_CIRCLE_MAX = 0.22
LIP_COLOR = (196, 64, 92)  # RGB, 로즈
LIP_MAX = 0.6  # 립 컬러 100%: 입술 색(밝기 제외)을 60%만큼 바꾼다
TEETH_WHITEN, TEETH_BRIGHT = 0.8, 0.15  # 치아 노란기를 80% 빼고, 밝기 여유의 15%만큼 밝게
EYE_SHARPEN = 1.5
BLEMISH_DEPTH = 8  # 주변(가운데값)보다 G 채널이 이만큼 넘게 어두우면 잡티로 보기 시작, 2배면 완전히
BLEMISH_KEEP_OUT = 0.12  # 눈·콧볼·입술 둘레 이만큼(눈 사이 거리 대비)은 잡티를 찾지 않는다
BLEMISH_ROUND = 2.5  # 긴 변이 짧은 변의 2.5배를 넘으면 잡티가 아니라 선(머리카락·주름)
BLEMISH_DARK_AROUND = 0.85  # 둘레(가운데값)가 얼굴 피부 전체의 85%보다 어두우면 잡티로 보지 않는다
BLEMISH_SIZE = 0.1  # 잡티 크기 한계 — 주변 가운데값을 이 지름(눈 사이 거리 대비)으로 잰다 (그보다 큰 그늘은 잡티가 아님)
SHADE = (0.10, 0.13, 0.17)  # 윤곽 쉐딩 100%: R·G·B를 이만큼 어둡게 (파랑을 더 빼서 따뜻한 그림자)
HIGHLIGHT_MAX = 0.16
EYESHADOW_MAX = 0.55  # 아이섀도 100%: 눈두덩 색(a·b)을 55%, 밝기를 그 절반만큼 색 쪽으로


def apply(rgb: np.ndarray, landmarks: np.ndarray | None, settings: BeautySettings, frame_width: int | None = None) -> np.ndarray:
    """보정한 새 RGB 배열. 얼굴이 없거나 효과가 다 0이면 원본 그대로 (복사 없이).

    옆얼굴·작은 얼굴은 효과를 줄인다 (정면·크기 가중치). rgb가 화면 일부(얼굴 주변)면 frame_width에 전체 폭을 준다.
    색(피부·메이크업)을 먼저 입히고 변형(얼굴형)은 마지막에: 색 영역이 랜드마크와 딱 맞고, 변형이 색칠한 피부를 같이 옮긴다.
    """
    if landmarks is None or settings.is_identity or np.isnan(landmarks).any():
        return rgb
    weight = frontal_weight(landmarks) * size_weight(landmarks, frame_width or rgb.shape[1])
    if weight <= 0:
        return rgb
    k = {name: getattr(settings, name) * weight for name in EFFECTS}
    out = rgb.copy()
    x0, y0, x1, y1 = face_box(landmarks, rgb.shape[1], rgb.shape[0])
    face, lm = out[y0:y1, x0:x1], landmarks - (x0, y0)  # face는 out의 일부 (바꾸면 out이 바뀐다)
    if k["blemish"] > 0:
        _blemish(face, lm, k["blemish"])  # 흐리기 전에 (흐리면 잡티가 번져 찾기 어렵다)
    _skin(face, lm, k["smooth"], k["brighten"])
    if k["dark_circle"] > 0:
        _lift(face, _under_eye_mask(face.shape, lm), DARK_CIRCLE_MAX * min(1.0, k["dark_circle"]))
    if k["contour"] > 0:
        _contour(face, lm, k["contour"])
    if k["blush"] > 0:
        for ellipse in cheek_ellipses(lm):
            _blush(face, ellipse, k["blush"])
    if k["eyeshadow"] > 0:
        _eyeshadow(face, lm, k["eyeshadow"], settings.eyeshadow_color)
    if k["lip"] > 0:
        _lips(face, lm, k["lip"])
    if k["teeth"] > 0:
        _teeth(face, lm, k["teeth"])
    if k["eye_clear"] > 0:
        _eye_clear(face, lm, k["eye_clear"])
    # 윤곽 변형(갸름·광대·턱)은 서로 붙어 있어 한 변형장으로 (따로 하면 이음매가 물결친다)
    _warp(face, slim_pulls(lm, k["slim"]) + cheekbone_pulls(lm, k["cheekbone"]) + chin_pulls(lm, k["chin"]))
    _warp(face, nose_pulls(lm, k["nose_slim"]))
    _warp(face, nose_length_pulls(lm, k["nose_length"]))
    _warp(face, smile_pulls(lm, k["smile"]))
    for zoom in eye_zooms(lm, k["eyes"]):
        _zoom(face, zoom)
    return out


def face_box(landmarks: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """효과가 닿을 수 있는 영역, 짝수 좌표 (색 정보가 2×2 단위라서).

    랜드마크 바깥으로 가장 멀리 닿는 효과는 턱선 당기기(반지름 = 눈 사이 거리 × SLIM_RADIUS) → 그보다 조금 넉넉히.
    """
    margin = eye_distance(landmarks) * (SLIM_RADIUS + 0.05)
    x0, y0 = landmarks.min(axis=0) - margin
    x1, y1 = landmarks.max(axis=0) + margin
    x0, y0 = max(0, int(x0) // 2 * 2), max(0, int(y0) // 2 * 2)
    x1, y1 = min(width, (int(x1) + 2) // 2 * 2), min(height, (int(y1) + 2) // 2 * 2)
    return x0, y0, x1, y1


def apply_frame(frame: av.VideoFrame, landmarks: np.ndarray | None, settings: BeautySettings) -> av.VideoFrame:
    """영상 프레임(YUV)에 보정을 입힌다. 얼굴 주변만 RGB로 바꿔 효과를 넣고, 효과가 바꾼 화소만 YUV로 되돌려 쓴다.

    화면 전체를 RGB로 바꿨다 되돌리면 효과가 없어도 PSNR 47dB까지 떨어졌다 (실측) — 효과가 닿지 않은 화소는
    얼굴 주변이라도 원래 YUV 값 그대로 둔다.
    """
    if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    if landmarks is None or settings.is_identity or np.isnan(landmarks).any():
        return frame
    w, h = frame.width, frame.height
    y_plane, u_plane, v_plane = planes(frame)  # 판을 따로 (높이 절반이 홀수여도 — services.yuv)
    x0, y0, x1, y1 = face_box(landmarks, w, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    cx0, cy0, cx1, cy1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
    roi = np.concatenate(
        [y_plane[y0:y1, x0:x1].ravel(), u_plane[cy0:cy1, cx0:cx1].ravel(), v_plane[cy0:cy1, cx0:cx1].ravel()]
    ).reshape((y1 - y0) * 3 // 2, x1 - x0)
    rgb = cv2.cvtColor(roi, cv2.COLOR_YUV2RGB_I420)
    after = apply(rgb, landmarks - (x0, y0), settings, frame_width=w)
    if after is rgb:
        return frame
    back = cv2.cvtColor(after, cv2.COLOR_RGB2YUV_I420).ravel()  # 같은 변환의 역이라 바뀐 곳과 안 바뀐 곳 경계가 티 나지 않는다
    n_y, n_c = (y1 - y0) * (x1 - x0), (cy1 - cy0) * (cx1 - cx0)
    changed = (after != rgb).any(axis=2)
    changed_c = changed.reshape(cy1 - cy0, 2, cx1 - cx0, 2).any(axis=(1, 3))  # 색 정보는 2×2 화소마다 하나
    y_plane[y0:y1, x0:x1][changed] = back[:n_y].reshape(y1 - y0, x1 - x0)[changed]
    u_plane[cy0:cy1, cx0:cx1][changed_c] = back[n_y : n_y + n_c].reshape(cy1 - cy0, cx1 - cx0)[changed_c]
    v_plane[cy0:cy1, cx0:cx1][changed_c] = back[n_y + n_c :].reshape(cy1 - cy0, cx1 - cx0)[changed_c]
    return from_planes(y_plane, u_plane, v_plane, frame)


REGION_COLORS = {  # RGB
    "eyeshadow": (190, 90, 255),  # 고른 색(브라운 등)으로 칠하면 피부와 비슷해 안 보인다
    "skin": (120, 230, 140),
    "blemish": (255, 80, 200),
    "smile": (255, 140, 200),
    "shade": (120, 70, 40),
    "highlight": (255, 250, 220),
    "dark_circle": (170, 140, 255),
    "slim": (255, 210, 60),
    "cheekbone": (255, 150, 40),
    "chin": (230, 120, 255),
    "eyes": (80, 160, 255),
    "nose_slim": (60, 200, 255),
    "nose_length": (40, 230, 220),
    "blush": (255, 105, 150),
    "lip": (255, 60, 90),
    "teeth": (255, 255, 255),
    "eye_clear": (255, 255, 120),
}
REGION_ALPHA = 0.4


def regions(rgb: np.ndarray, landmarks: np.ndarray | None, settings: BeautySettings) -> np.ndarray:
    """켠 효과가 들어갈 자리를 원본 위에 반투명하게 칠한 그림 (효과마다 색), 변형은 움직임 화살표도.

    효과마다 영역을 한 번에 칠하고 한 번만 섞는다 (원이 겹쳐도 진해지지 않게). 끈 효과는 그리지 않는다 (13가지라 다 그리면 어지럽다).
    """
    out = rgb.copy()
    if landmarks is None or np.isnan(landmarks).any():
        return out
    x0, y0, x1, y1 = face_box(landmarks, rgb.shape[1], rgb.shape[0])
    face, lm = out[y0:y1, x0:x1], landmarks - (x0, y0)
    thick = max(1, round(eye_distance(lm) / 70))  # 화면 크기가 아니라 얼굴 크기 기준 (4K에서 너무 두꺼웠다)
    arrows = []
    for kind, mask, pulls in _areas(face, lm, settings):
        alpha = (REGION_ALPHA * mask)[..., None]
        face[:] = (face + (np.array(REGION_COLORS[kind], np.float32) - face) * alpha).astype(np.uint8)
        arrows += [(kind, pull) for pull in pulls]
    for kind, pull in arrows:
        # 화살표는 실제 움직임보다 3배 길게 (몇 px라 그대로면 안 보인다)
        tip = (round(pull.cx + (pull.tx - pull.cx) * 3), round(pull.cy + (pull.ty - pull.cy) * 3))
        cv2.arrowedLine(face, (round(pull.cx), round(pull.cy)), tip, REGION_COLORS[kind], thick, cv2.LINE_AA, tipLength=0.35)
    return out


def _areas(image: np.ndarray, lm: np.ndarray, settings: BeautySettings) -> list[tuple[str, np.ndarray, list[Pull]]]:
    """regions가 칠할 (효과, 영역 0~1, 화살표) — 켠 효과만. 잡티는 찾은 점을 칠한다."""
    shape = image.shape
    def circles(shapes) -> np.ndarray:
        mask = np.zeros(shape[:2], np.float32)
        for c in shapes:
            cv2.circle(mask, (round(c.cx), round(c.cy)), round(c.radius), 1.0, -1, cv2.LINE_AA)
        return mask

    areas = []
    if settings.smooth or settings.brighten:
        areas.append(("skin", _skin_mask(shape, lm), []))
    if settings.dark_circle:
        areas.append(("dark_circle", np.minimum(1.0, 2 * _under_eye_mask(shape, lm)), []))
    if settings.blemish:
        areas.append(("blemish", np.minimum(1.0, 3 * _blemish_spots(image, lm)), []))
    for kind, make in (("slim", slim_pulls), ("cheekbone", cheekbone_pulls), ("chin", chin_pulls),
                       ("nose_slim", nose_pulls), ("nose_length", nose_length_pulls), ("smile", smile_pulls)):
        pulls = make(lm, getattr(settings, kind))
        if pulls:
            areas.append((kind, circles(pulls), pulls))
    if settings.eyes:
        areas.append(("eyes", circles(eye_zooms(lm, settings.eyes)), []))
    if settings.blush:
        mask = np.zeros(shape[:2], np.float32)
        for e in cheek_ellipses(lm):
            center, axes = (round(e.cx), round(e.cy)), (round(e.rx), round(e.ry))
            cv2.ellipse(mask, center, axes, math.degrees(e.angle), 0, 360, 1.0, -1, cv2.LINE_AA)
        areas.append(("blush", mask, []))
    if settings.contour:
        shade, highlight = contour_shapes(lm)
        areas.append(("shade", np.minimum(1.0, 2 * _ellipses_mask(shape, shade)), []))
        areas.append(("highlight", np.minimum(1.0, 2 * _ellipses_mask(shape, highlight)), []))
    if settings.eyeshadow:
        areas.append(("eyeshadow", np.minimum(1.0, 1.5 * _eyelid_mask(shape, lm)), []))
    if settings.lip:
        areas.append(("lip", _lip_mask(shape, lm), []))
    if settings.teeth:
        areas.append(("teeth", _fill(shape, [lip_polygons(lm)[1]]), []))
    if settings.eye_clear:
        areas.append(("eye_clear", _eye_mask(shape, lm), []))
    return areas


def _roi(image: np.ndarray, cx: float, cy: float, radius: float) -> tuple[int, int, int, int] | None:
    h, w = image.shape[:2]
    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1, y1 = min(w, int(cx + radius) + 1), min(h, int(cy + radius) + 1)
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _warp(image: np.ndarray, pulls: list[Pull]) -> None:
    """여러 점을 한 번에 당기는 매끄러운 변형. 제자리에서 바꾼다.

    점마다 차례로 변형하면 이웃 변형끼리 겹쳐 턱선이 물결쳤다 (실측: 갸름 80%) →
    모든 점의 움직임을 하나의 변형장으로 합친다: 점 가까이에선 그 점의 움직임, 멀어질수록 부드럽게 0
    (가중 평균 × 가까운 정도, 가중치 (1 - (거리/반지름)²)²).
    """
    if not pulls:
        return
    left = min(p.cx - p.radius for p in pulls)
    top = min(p.cy - p.radius for p in pulls)
    right = max(p.cx + p.radius for p in pulls)
    bottom = max(p.cy + p.radius for p in pulls)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, int(left)), max(0, int(top)), min(w, int(right) + 1), min(h, int(bottom) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    total = np.zeros_like(xs)
    move_x, move_y = np.zeros_like(xs), np.zeros_like(xs)
    for pull in pulls:
        box = _roi(image, pull.cx, pull.cy, pull.radius)  # 점마다 자기 반지름 안만 계산 (전체 영역을 돌면 몇 배 느리다)
        if box is None:
            continue
        sy, sx = slice(box[1] - y0, box[3] - y0), slice(box[0] - x0, box[2] - x0)
        t2 = ((xs[sy, sx] - pull.cx) ** 2 + (ys[sy, sx] - pull.cy) ** 2) / pull.radius**2
        weight = np.clip(1 - t2, 0, None) ** 2
        total[sy, sx] += weight
        move_x[sy, sx] += weight * (pull.tx - pull.cx)
        move_y[sy, sx] += weight * (pull.ty - pull.cy)
    # 가중 평균 움직임 × 가까운 정도(겹친 곳에서 움직임이 더해져 커지지 않게, 가장자리는 0으로)
    scale = np.minimum(total, 1.0) / np.maximum(total, 1e-6)
    # 결과의 (x, y) 화소는 원본의 (x - 움직임) 에서 가져온다 → 내용이 움직임 방향으로 밀린다
    map_x, map_y = xs - move_x * scale, ys - move_y * scale
    image[y0:y1, x0:x1] = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _blush(image: np.ndarray, ellipse: Ellipse, strength: float) -> None:
    """볼 사과 자리에 가장자리로 갈수록 옅어지는 타원으로 색을 섞는다."""
    box = _roi(image, ellipse.cx, ellipse.cy, 1.6 * max(ellipse.rx, ellipse.ry))
    if box is None:
        return
    x0, y0, x1, y1 = box
    alpha = (_ellipse_falloff(ellipse, x0, y0, x1, y1) * (BLUSH_MAX_ALPHA * min(1.0, strength)))[..., None]
    region = image[y0:y1, x0:x1].astype(np.float32)
    color = np.array(BLUSH_COLOR, np.float32)
    image[y0:y1, x0:x1] = (region + (color - region) * alpha).astype(np.uint8)


def _ellipse_falloff(ellipse: Ellipse, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """타원 가운데 1, 바깥으로 부드럽게 0 (타원 경계에서 약 0.14)."""
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    c, s = math.cos(ellipse.angle), math.sin(ellipse.angle)
    u = ((xs - ellipse.cx) * c + (ys - ellipse.cy) * s) / ellipse.rx
    v = (-(xs - ellipse.cx) * s + (ys - ellipse.cy) * c) / ellipse.ry
    return np.exp(-2.0 * (u * u + v * v))


def _fill(shape, polygons: list[np.ndarray], feather: float = 0.0) -> np.ndarray:
    """다각형들을 칠한 영역 (0~1). feather(px)만큼 가장자리를 부드럽게."""
    mask = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polygons], 1.0, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), feather) if feather > 0 else mask


def _grow(mask: np.ndarray, px: float) -> np.ndarray:
    size = max(1, round(px)) * 2 + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def _soft(shape, lm: np.ndarray, make) -> np.ndarray:
    """가장자리가 부드러운 큰 영역은 줄인 크기(눈 사이 SMOOTH_EYES_PX)에서 만들어 늘린다 — 모양은 같고 몇 배 빠르다."""
    scale = min(1.0, SMOOTH_EYES_PX / max(eye_distance(lm), 1.0))
    if scale >= 1.0:
        return make(shape[:2], lm)
    h, w = shape[:2]
    small = make((max(1, round(h * scale)), max(1, round(w * scale))), lm * scale)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _box(mask: np.ndarray) -> tuple[slice, slice] | None:
    """mask가 0이 아닌 곳을 감싸는 영역."""
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return None
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def _skin_mask(shape, lm: np.ndarray) -> np.ndarray:
    """피부만: 얼굴 윤곽 안(이마 아래쪽까지)에서 눈·눈썹·입술을 넉넉히 뺀 영역, 가장자리 부드럽게."""
    def make(shape, lm):
        eyes = eye_distance(lm)
        holes = _grow(_fill(shape, eye_polygons(lm) + brow_polygons(lm) + [lip_polygons(lm)[0]]), 0.06 * eyes)
        return cv2.GaussianBlur(_fill(shape, [skin_polygon(lm)]) * (1 - holes), (0, 0), 0.04 * eyes)

    return _soft(shape, lm, make)


def _eye_mask(shape, lm: np.ndarray) -> np.ndarray:
    eyes = eye_distance(lm)
    return cv2.GaussianBlur(_grow(_fill(shape, eye_polygons(lm)), 0.02 * eyes), (0, 0), 0.015 * eyes)


def _lip_mask(shape, lm: np.ndarray) -> np.ndarray:
    outer, inner = lip_polygons(lm)
    return cv2.GaussianBlur(_fill(shape, [outer]) * (1 - _fill(shape, [inner])), (0, 0), 0.01 * eye_distance(lm))


def _under_eye_mask(shape, lm: np.ndarray) -> np.ndarray:
    """눈 밑 타원 (가운데 1, 부드럽게 0) — 눈 자체는 뺀다."""
    def make(shape, lm):
        mask = np.zeros(shape[:2], np.float32)
        for ellipse in under_eye_ellipses(lm):
            mask = np.maximum(mask, _ellipse_falloff(ellipse, 0, 0, shape[1], shape[0]))
        return mask * (1 - _grow(_fill(shape, eye_polygons(lm)), 0.04 * eye_distance(lm)))

    return _soft(shape, lm, make)


def _skin(face: np.ndarray, lm: np.ndarray, smooth: float, brighten: float) -> None:
    """피부 매끈하게(가장자리는 지키는 흐리기) · 톤 밝게. 제자리에서."""
    if smooth <= 0 and brighten <= 0:
        return
    mask = _skin_mask(face.shape, lm)
    box = _box(mask)
    if box is None:
        return
    region, mask = face[box], mask[box][..., None]
    work = region.astype(np.float32)
    if smooth > 0:
        h, w = region.shape[:2]
        scale = min(1.0, SMOOTH_EYES_PX / max(eye_distance(lm), 1.0))
        small = cv2.resize(region, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        for _ in range(2):  # 한 번은 잡티가 남는다
            small = cv2.bilateralFilter(small, 7, 30, 5)
        soft = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        work += (soft - work) * mask * (SMOOTH_MAX * min(1.0, smooth))
    if brighten > 0:
        work += (255 - work) * mask * (BRIGHTEN_MAX * min(1.0, brighten))
    region[:] = np.clip(work + 0.5, 0, 255).astype(np.uint8)


def _lift(face: np.ndarray, mask: np.ndarray, amount: float) -> None:
    """mask 자리를 남은 밝기 여유의 amount만큼 밝게."""
    box = _box(mask)
    if box is None:
        return
    work = face[box].astype(np.float32)
    work += (255 - work) * (mask[box] * amount)[..., None]
    face[box] = np.clip(work + 0.5, 0, 255).astype(np.uint8)


def _in_lab(face: np.ndarray, alpha: np.ndarray, change) -> None:
    """alpha > 0인 자리만 Lab(실수: L 0~100, a·b ±127)으로 바꿔 change(lab, alpha)를 하고 되돌린다.

    정수 Lab 왕복은 화소값이 조금씩 틀어진다 → 실수로, 그리고 영역 밖은 원래 값 그대로 둔다.
    """
    box = _box(alpha)
    if box is None:
        return
    crop, a = face[box], alpha[box]
    lab = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    change(lab, a)
    back = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255 + 0.5, 0, 255).astype(np.uint8)
    crop[:] = np.where((a > 1e-3)[..., None], back, crop)


def _lips(face: np.ndarray, lm: np.ndarray, strength: float) -> None:
    """입술 색(a·b)만 립 컬러 쪽으로 — 밝기(L)는 그대로 둬서 입술 결·윤기가 남는다."""
    target = cv2.cvtColor(np.array([[LIP_COLOR]], np.float32) / 255, cv2.COLOR_RGB2LAB)[0, 0]

    def tint(lab, a):
        lab[..., 1:] += (target[1:] - lab[..., 1:]) * (a * LIP_MAX * min(1.0, strength))[..., None]

    _in_lab(face, _lip_mask(face.shape, lm), tint)


def _teeth(face: np.ndarray, lm: np.ndarray, strength: float) -> None:
    """입 안쪽에서 밝은 화소(치아)만 노란기(b)를 빼고 조금 밝게. 어두운 입속·혀는 그대로."""
    inner = _fill(face.shape, [lip_polygons(lm)[1]], 0.005 * eye_distance(lm))

    def whiten(lab, a):
        w = a * np.clip((lab[..., 0] - 45) / 20, 0, 1) * min(1.0, strength)  # L 45 아래(입속 그늘)는 0, 65 위는 1
        lab[..., 2] -= lab[..., 2] * TEETH_WHITEN * w
        lab[..., 1] -= lab[..., 1] * 0.3 * w
        lab[..., 0] += (100 - lab[..., 0]) * TEETH_BRIGHT * w

    _in_lab(face, inner, whiten)


def _eye_clear(face: np.ndarray, lm: np.ndarray, strength: float) -> None:
    """눈 안(흰자·눈동자·속눈썹)을 선명하게 (언샤프 마스크)."""
    mask = _eye_mask(face.shape, lm)
    box = _box(mask)
    if box is None:
        return
    work = face[box].astype(np.float32)
    blur = cv2.GaussianBlur(work, (0, 0), max(1.0, 0.012 * eye_distance(lm)))
    work += (work - blur) * EYE_SHARPEN * mask[box][..., None] * min(1.0, strength)
    face[box] = np.clip(work + 0.5, 0, 255).astype(np.uint8)


def _ellipses_mask(shape, ellipses: list[Ellipse]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.float32)
    for ellipse in ellipses:
        box = _roi(mask, ellipse.cx, ellipse.cy, 1.6 * max(ellipse.rx, ellipse.ry))
        if box is not None:
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], _ellipse_falloff(ellipse, x0, y0, x1, y1))
    return mask


def _blemish_parts(face: np.ndarray, lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(잡티 자리 0~1, 둘레 피부 = 가운데값 RGB) — 둘 다 줄인 크기(눈 사이 SMOOTH_EYES_PX)에서. 찾기와 메꾸기가 같은 가운데값을 쓴다.

    잡티 = 피부 안에서 주변 가운데값보다 작고 어두운 점. G 채널로 본다 — 갈색 점과 붉은 뾰루지가 둘 다 G에서 어둡다.
    0/1로 끊지 않고 깊이에 따라 서서히 (경계에 걸친 점이 프레임마다 켜졌다 꺼지며 깜빡이지 않게).
    실측에서 잡티로 잘못 잡힌 곳 — 눈꼬리 속눈썹 그늘, 콧볼 옆 주름·콧구멍, 입꼬리, 얼굴 가장자리(배경),
    머리카락 — 은 뺀다.
    """
    h, w = face.shape[:2]
    scale = min(1.0, SMOOTH_EYES_PX / max(eye_distance(lm), 1.0))
    small = cv2.resize(face, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    lm, eyes = lm * scale, eye_distance(lm) * scale
    skin = np.clip(_skin_mask(small.shape, lm) * 2 - 1, 0, 1)  # 가장자리 흐린 부분은 빼고 안쪽만
    features = _fill(small.shape, eye_polygons(lm) + [nostril_polygon(lm), lip_polygons(lm)[0]])
    area = skin * (1 - _grow(features, BLEMISH_KEEP_OUT * eyes))
    size = max(3, round(BLEMISH_SIZE * eyes) // 2 * 2 + 1)
    base = cv2.medianBlur(small, size)
    g, around = small[..., 1].astype(np.float32), base[..., 1].astype(np.float32)
    skin_level = float(np.median(around[area > 0.5])) if (area > 0.5).any() else 255.0
    # 둘레가 피부보다 확 어두우면 머리카락이 섞인 곳(머리카락선·앞머리) → 잡티를 찾지 않는다
    spots = np.clip((around - g - BLEMISH_DEPTH) / BLEMISH_DEPTH, 0, 1) * area * (around > BLEMISH_DARK_AROUND * skin_level)
    # 가늘고 긴 것(머리카락·주름·그늘 가장자리)은 잡티가 아니다 → 작고 둥근 덩어리만 남긴다
    _, labels, stats, _ = cv2.connectedComponentsWithStats((spots > 0).astype(np.uint8), connectivity=8)
    long_side = np.maximum(stats[:, 2], stats[:, 3])
    keep = (long_side <= size) & (long_side <= BLEMISH_ROUND * np.minimum(stats[:, 2], stats[:, 3]))
    keep[0] = False  # 배경
    spots *= keep[labels]
    spots = cv2.GaussianBlur(cv2.dilate(spots, np.ones((3, 3), np.uint8)), (0, 0), 1.0) * area  # 점 가장자리까지 덮게
    return spots, base


def _blemish_spots(face: np.ndarray, lm: np.ndarray) -> np.ndarray:
    spots, _ = _blemish_parts(face, lm)
    return cv2.resize(spots, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LINEAR)


def _blemish(face: np.ndarray, lm: np.ndarray, strength: float) -> None:
    """잡티 자리를 둘레 피부(가운데값)로 메꾼다."""
    spots, base = _blemish_parts(face, lm)
    size = (face.shape[1], face.shape[0])
    spots = cv2.resize(spots, size, interpolation=cv2.INTER_LINEAR)
    box = _box(spots)
    if box is None:
        return
    fill = cv2.resize(base, size, interpolation=cv2.INTER_LINEAR)[box].astype(np.float32)
    region = face[box]
    alpha = (spots[box] * min(1.0, strength))[..., None]
    region[:] = np.clip(region + (fill - region) * alpha + 0.5, 0, 255).astype(np.uint8)


def _contour(face: np.ndarray, lm: np.ndarray, strength: float) -> None:
    """쉐딩 자리는 따뜻하게 어둡게, 하이라이트 자리는 밝게. 피부 영역 안에서만 (영역은 줄인 크기에서 만든다)."""
    dark = _soft(face.shape, lm, lambda shape, l: _ellipses_mask(shape, contour_shapes(l)[0]) * _skin_mask(shape, l))
    light = _soft(face.shape, lm, lambda shape, l: _ellipses_mask(shape, contour_shapes(l)[1]) * _skin_mask(shape, l))
    k = min(1.0, strength)
    box = _box(dark)  # 쉐딩(양 볼·코 옆)과 하이라이트(가운데 세로줄)는 따로 — 합친 상자는 얼굴 전체라 느리다
    if box is not None:
        work = face[box].astype(np.float32)
        work *= 1 - dark[box][..., None] * (np.array(SHADE, np.float32) * k)
        face[box] = np.clip(work + 0.5, 0, 255).astype(np.uint8)
    box = _box(light)
    if box is not None:
        work = face[box].astype(np.float32)
        work += (255 - work) * (light[box] * HIGHLIGHT_MAX * k)[..., None]
        face[box] = np.clip(work + 0.5, 0, 255).astype(np.uint8)


def _eyelid_mask(shape, lm: np.ndarray) -> np.ndarray:
    eyes = eye_distance(lm)
    lids = _fill(shape, eyelid_polygons(lm)) * (1 - _grow(_fill(shape, eye_polygons(lm)), 0.01 * eyes))
    return cv2.GaussianBlur(lids, (0, 0), 0.03 * eyes)


def _eyeshadow(face: np.ndarray, lm: np.ndarray, strength: float, color: str) -> None:
    """눈두덩에 색: 색(a·b)은 많이, 밝기는 조금만 색 쪽으로 — 피부 결이 남는다."""
    rgb = EYESHADOW_COLORS[color if color in EYESHADOW_COLORS else "brown"][1]
    target = cv2.cvtColor(np.array([[rgb]], np.float32) / 255, cv2.COLOR_RGB2LAB)[0, 0]

    def tint(lab, a):
        amount = (a * EYESHADOW_MAX * min(1.0, strength))[..., None]
        lab[..., 1:] += (target[1:] - lab[..., 1:]) * amount
        lab[..., :1] += (target[:1] - lab[..., :1]) * amount * 0.5

    _in_lab(face, _eyelid_mask(face.shape, lm), tint)


def _zoom(image: np.ndarray, zoom: Zoom) -> None:
    """(cx, cy) 둘레를 확대: 가운데에서 1/(1-amount)배, 반지름에서 1배 (경계에서 기울기까지 이어져 티가 안 난다)."""
    box = _roi(image, zoom.cx, zoom.cy, zoom.radius)
    if box is None:
        return
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - zoom.cx, ys - zoom.cy
    t2 = (dx * dx + dy * dy) / zoom.radius**2
    factor = 1 - zoom.amount * np.clip(1 - t2, 0, None) ** 2  # 결과 화소는 가운데 쪽 원본에서 가져온다 → 확대
    image[y0:y1, x0:x1] = cv2.remap(image, zoom.cx + dx * factor, zoom.cy + dy * factor, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)


def render_beautified(path: Path, output: Path, track: np.ndarray, settings: BeautySettings, on_progress=None) -> None:
    """path(고정 fps 파일)를 보정해서 output으로. track = smooth_track을 거친 프레임별 랜드마크. 소리는 그대로 복사."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    try:
        with av.open(str(path)) as src, av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            v_in = src.streams.video[0]
            v_in.thread_type = "AUTO"
            a_in = src.streams.audio[0] if src.streams.audio else None
            fps = float(v_in.average_rate)
            size = (v_in.codec_context.width, v_in.codec_context.height)
            v_out = add_video_stream(dst, size, v_in.average_rate)
            frame_tb = 1 / Fraction(v_in.average_rate)
            a_out = dst.add_stream_from_template(a_in) if a_in is not None else None
            total = len(track)
            index = 0
            for packet in src.demux(*(s for s in (v_in, a_in) if s is not None)):
                if packet.stream is a_in:
                    if packet.dts is not None:  # 소리는 그대로 옮긴다 (끝 표시 빈 패킷은 버림)
                        packet.stream = a_out
                        dst.mux(packet)
                    continue
                for decoded in packet.decode():  # 영상은 끝 표시 패킷도 디코딩해야 남은 프레임이 나온다
                    yield_to_playback()  # 재생 중이면 잠깐 멈춘다
                    landmarks = track[index] if index < total else None
                    frame = apply_frame(decoded, landmarks, settings)
                    # v_out.time_base는 첫 mux 때 컨테이너가 1/15360 같은 값으로 바꾼다 → 고정된 1/fps를 써야 한다
                    frame.pts, frame.time_base = index, frame_tb
                    dst.mux(v_out.encode(frame))
                    index += 1
                    if on_progress and total and index % 15 == 0:
                        on_progress(min(1.0, index / total))
            dst.mux(v_out.encode(None))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)


CLIP_SECONDS = 60  # 미리보기 영상 길이
CLIP_HALF = (960, 1080)  # 미리보기 영상 한 쪽(원본 또는 보정)의 최대 크기
CLIP_GAP = 8  # 두 쪽 사이 검은 틈


def _even(value: float) -> int:
    return max(2, round(value) // 2 * 2)


def render_side_by_side(
    path: Path, output: Path, start: int, count: int, landmarks, settings: BeautySettings, on_progress=None
) -> None:
    """path(고정 fps 파일)의 start번째 프레임부터 count장을 '왼쪽 원본 | 오른쪽 보정' 한 영상으로 (소리는 그 구간).

    두 영상을 따로 틀면 재생이 조금씩 어긋난다 → 한 파일로 붙여서 언제나 같은 순간을 비교한다.
    landmarks(i) = i번째(구간 안 0부터) 프레임의 랜드마크 또는 None.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    try:
        with av.open(str(path)) as probe:
            v_in = probe.streams.video[0]
            rate, width, height = v_in.average_rate, v_in.codec_context.width, v_in.codec_context.height
            has_audio = bool(probe.streams.audio)
        scale = min(CLIP_HALF[0] / width, CLIP_HALF[1] / height, 1.0)
        half_w, half_h = _even(width * scale), _even(height * scale)
        size = (2 * half_w + CLIP_GAP, half_h)
        frame_tb = 1 / Fraction(rate)
        with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            v_out = add_video_stream(dst, size, rate)
            a_out = dst.add_stream("aac", rate=AUDIO_RATE, layout="stereo") if has_audio else None
            canvas = np.zeros((half_h, size[0], 3), np.uint8)
            done = 0
            for _index, frame in frames_from(path, start):
                if done >= count:
                    break
                yield_to_playback()
                after = apply_frame(frame, landmarks(done), settings)
                canvas[:, :half_w] = frame.reformat(half_w, half_h, format="rgb24").to_ndarray()
                canvas[:, half_w + CLIP_GAP :] = after.reformat(half_w, half_h, format="rgb24").to_ndarray()
                out = av.VideoFrame.from_ndarray(canvas, format="rgb24").reformat(format="yuv420p")
                out.pts, out.time_base = done, frame_tb
                dst.mux(v_out.encode(out))
                done += 1
                if on_progress and done % 15 == 0:
                    on_progress(min(1.0, done / count))
            dst.mux(v_out.encode(None))
            if a_out is not None:
                _audio_range(path, dst, a_out, start * float(frame_tb), done * float(frame_tb))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)


def _audio_range(path: Path, dst, a_out, start: float, duration: float) -> None:
    """path의 소리 start초부터 duration초를 a_out(48kHz 스테레오 AAC)으로."""
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
    first, total = round(start * AUDIO_RATE), round(duration * AUDIO_RATE)
    written = 0
    with av.open(str(path)) as src:
        stream = src.streams.audio[0]
        src.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream, backward=True)
        for frame in src.decode(stream):
            begin = round((frame.time or 0) * AUDIO_RATE)  # 이 덩어리 첫 샘플의 위치 (원본 기준)
            for resampled in resampler.resample(frame):
                samples = resampled.to_ndarray()
                lo = max(0, first - begin)
                hi = min(samples.shape[1], first + total - begin)
                begin += samples.shape[1]
                if hi <= lo:
                    continue
                chunk = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:, lo:hi]), format="fltp", layout="stereo")
                chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, written, Fraction(1, AUDIO_RATE)
                written += hi - lo
                dst.mux(a_out.encode(chunk))
            if written >= total:
                break
    dst.mux(a_out.encode(None))


class AiWorker(QObject):
    """AI 작업을 스레드 하나에서 차례로. 미리보기는 '가장 최근 요청'만 처리한다 (스크럽 중 밀린 요청은 버림)."""

    samples_ready = Signal(Path, object)  # 원본, list[Probe] (샘플 후보 장면과 얼굴들)
    preview_ready = Signal(Path, int, object, object)  # 원본, 프레임, 원래 RGB, 랜드마크(없으면 None)
    progress = Signal(str, Path, float)  # "samples"·"analyze"·"render"·"clip", 원본, 0~1
    rendered = Signal(Path, Path)  # 원본, 보정본
    clip_ready = Signal(Path, Path)  # 원본, 원본|보정 나란히 미리보기 영상
    failed = Signal(str, Path, str)  # 작업 종류, 원본, 이유

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._detector = FaceDetector()
        self._jobs: queue.Queue = queue.Queue()
        self._preview: tuple | None = None
        self._lock = threading.Lock()
        self.busy_with: Path | None = None  # 렌더 중인 원본
        threading.Thread(target=self._worker, daemon=True, name="ai").start()

    def request_samples(self, path: Path, cache: Path | None = None, refresh: bool = False) -> None:
        """cache가 있으면 거기서 바로 (refresh면 다시 찾아 덮어쓴다)."""
        self._jobs.put(("samples", path, cache, refresh))

    def request_preview(self, path: Path, frame: int, track_path: Path, target) -> None:
        with self._lock:
            self._preview = (path, frame, track_path, target)
        self._jobs.put(("preview",))

    def request_render(self, path: Path, output: Path, track_path: Path, settings: BeautySettings, original=None) -> None:
        """path = 얼굴을 분석한 파일(휴대폰 영상이면 편집용 사본). original = (원본, Plan)을 주면 사본 대신 원본에서
        회전·색 변환과 보정을 한 번에 해서 압축을 한 번 덜 한다."""
        self.busy_with = path
        self._jobs.put(("render", path, output, track_path, settings, original))

    def request_clip(self, path: Path, output: Path, start: int, track_path: Path, settings: BeautySettings) -> None:
        """start번째 프레임부터 약 CLIP_SECONDS초를 원본|보정 나란히 한 영상으로."""
        self._jobs.put(("clip", path, output, start, track_path, settings))

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            kind = job[0]
            try:
                if kind == "samples":
                    path, cache, refresh = job[1:]
                    probes = load_probes(cache) if cache is not None and cache.exists() and not refresh else None
                    if probes is None:
                        probes = scan(self._detector, path, lambda done, total: self.progress.emit("samples", path, done / total))
                        if cache is not None:
                            save_probes(cache, probes)
                    self.samples_ready.emit(path, probes)
                elif kind == "preview":
                    with self._lock:
                        request, self._preview = self._preview, None
                    if request is not None:
                        self._do_preview(*request)
                elif kind == "render":
                    self._do_render(*job[1:])
                elif kind == "clip":
                    self._do_clip(*job[1:])
            except Exception as e:
                log.exception("AI 작업 실패: %s", kind)
                path = job[1] if len(job) > 1 else Path()
                if kind == "render":
                    self.busy_with = None
                self.failed.emit(kind, path, str(e))

    def _do_preview(self, path: Path, frame: int, track_path: Path, target) -> None:
        rgb = frame_at(path, frame)
        if rgb is None:
            return
        landmarks = None
        if track_path.exists():  # 분석을 해 두었으면 떨림을 줄인 값을 쓴다 (보정본과 같은 모습)
            track = smooth_track(np.load(track_path))
            landmarks = track[frame] if frame < len(track) and not np.isnan(track[frame]).any() else None
        else:
            face = choose(self._detector.faces(rgb), target)
            landmarks = face.landmarks if face is not None else None
        self.preview_ready.emit(path, frame, rgb, landmarks)

    def _do_render(self, path: Path, output: Path, track_path: Path, settings: BeautySettings, original=None) -> None:
        if not track_path.exists():  # 처음이면 영상 전체 얼굴 분석부터 (강도를 바꿔도 다시 할 필요 없다)
            raw = analyze_video(self._detector, path, settings.target_vector, lambda r: self.progress.emit("analyze", path, r))
            track_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(track_path, raw.astype(np.float32))
        track = smooth_track(np.load(track_path).astype(np.float64))

        def on_progress(ratio: float) -> None:
            self.progress.emit("render", path, ratio)

        if original is not None:  # 원본 → (회전·색 변환 + 보정) → 압축 한 번. 프레임 번호는 사본과 같다 (같은 변환)
            source, plan = original

            def beautify(frame: av.VideoFrame) -> av.VideoFrame:
                index = frame.pts
                return apply_frame(frame, track[index] if 0 <= index < len(track) else None, settings)

            make_optimized(source, output, plan, on_progress, process=beautify)
        else:
            render_beautified(path, output, track, settings, on_progress)
        self.busy_with = None
        self.rendered.emit(path, output)


    def _do_clip(self, path: Path, output: Path, start: int, track_path: Path, settings: BeautySettings) -> None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            count = min(round(CLIP_SECONDS * float(stream.average_rate)), max(1, (stream.frames or 0) - start))
        if track_path.exists():  # 영상 전체를 분석해 두었으면 그대로 (보정본과 같은 모습)
            track = smooth_track(np.load(track_path).astype(np.float64))[start : start + count]
            share = 0.0
        else:  # 아니면 이 구간만 분석 (1분이면 영상 전체보다 훨씬 빠르다)
            raw = analyze_video(self._detector, path, settings.target_vector,
                                lambda r: self.progress.emit("clip", path, 0.5 * r), start=start, count=count)
            track = smooth_track(raw)
            share = 0.5

        def landmarks(i: int):
            return track[i] if i < len(track) and not np.isnan(track[i]).any() else None

        render_side_by_side(path, output, start, count, landmarks, settings,
                            lambda r: self.progress.emit("clip", path, share + (1 - share) * r))
        self.clip_ready.emit(path, output)


def track_path_for(cache_dir: Path, media: Path, settings: BeautySettings) -> Path:
    """영상 전체 얼굴 분석 결과를 둘 캐시 파일 (원본이나 보정할 사람이 바뀌면 이름도 바뀐다)."""
    return track_file(cache_dir, cache_name(media), settings.target_vector)
