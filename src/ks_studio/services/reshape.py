"""사람 층만 변형하고 배경은 그대로 — 몸매(허리·어깨·다리)와 얼굴 작게.

예전 '얼굴 작게'는 얼굴 점 106개로 만든 윤곽만 줄였는데, 그 윤곽이 실제 얼굴 경계보다 안쪽이라 원래 경계가 남아
이중 윤곽이 보였다 (사용 불가). 이제는:
  1. 사람 분리(services.body)로 사람 층(머리카락·피부·옷)을 얻는다 — 실제 경계
  2. 변형(어디를 얼마나 옮길지)을 화면 전체에 계산하고 사람 층과 그 투명도를 같이 옮긴다
  3. 사람이 비켜서 드러난 배경은 둘레 배경으로 메꾼 그림(inpaint)을 깔고, 그 위에 옮긴 사람 층을 얹는다
  4. 변형이 닿지 않은 곳은 원래 화소 그대로 — 배경은 휘지 않는다
몸의 점(RTMPose)은 허리·어깨·엉덩이 위치를, 허리·어깨의 실제 옷 경계는 사람 분리에서 찾는다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ks_studio.core.beauty import face_small_scale
from ks_studio.services.body import BACKGROUND, FACE_SKIN, HAIR, KEYPOINT_MIN_SCORE, PARTS_SIZE, BodyAnalyzer, layer, person_box

GRID = 4  # 변형은 4화소마다 계산하고 늘린다 (부드러운 변형이라 충분, 4K에서 16배 빠르다)
FILL_SCALE = 0.125  # 드러난 배경 메꾸기는 1/8 크기에서 (사람 뒤 흐린 배경이라 충분하고, 1/4보다 몇 배 빠르다)
HOLE_GROW = 1.5  # 메꿀 자리 = 사람 자리 + 사람 분리 한 칸(256 해상도)의 1.5배 — 분리 경계 밖에 남은 머리카락 끝이 '유령'처럼 보였다
POSE_LONG_SIDE = 960  # 몸의 점은 긴 변 960px로 줄인 그림에서 찾는다
WAIST_MAX = 0.12  # 강도 100%: 허리 양쪽 가장자리가 반폭의 12%만큼 안쪽으로
ARM_CLEAR = 0.3  # 허리 가장자리에서 몸통 길이의 30% 안에 팔꿈치·손목이 있으면 그쪽 허리는 건너뛴다
SHOULDER_MAX = 0.07  # 강도 100%: 어깨 끝이 어깨 너비의 7%만큼 안쪽으로
LEGS_MAX = 0.10  # 강도 100%: 엉덩이 아래가 10% 길어진다
HEAD_REACH = 2.0  # 머리카락은 얼굴 점이 닿는 거리의 2배 안쪽까지만 머리로 친다
HEAD_FADE = 1.6  # 머리 바깥 고리: 머리 반지름의 1.6배에서 변형이 0
BODY_EFFECTS = ("waist", "shoulders", "legs")
PERSON = tuple(i for i in range(6) if i != BACKGROUND)  # 사람 층 = 배경 아닌 것 전부 (머리카락·피부·옷·기타)
SMOOTH = 0.6  # 몸의 점 떨림 줄이기: 새 값 60% + 이전 40%
HEAD_MARGIN = 0.35  # 머리 둘레 다시 분리: 머리 상자를 사방으로 35% 넓혀서
HEAD_FEATHER = 0.15  # 머리 둘레 분리를 전체 분리에 이을 때, 상자 가장자리 15%에서 서서히 (이음매가 안 보이게)


@dataclass(frozen=True)
class BodyInfo:
    parts: np.ndarray  # (256, 256, 6) 부위 확률 (화면 전체)
    keypoints: np.ndarray | None  # (17, 3) 화면 좌표 (x, y, 점수)
    head: tuple | None = None  # (부위 확률, (x0, y0, x1, y1) 화면 좌표) — 머리 둘레만 다시 분리한 것 (귀걸이·잔머리까지)
    fill: object = None  # 사람 뒤 배경 메꾸기 (services.inpaint, 없으면 fill_telea)


def needs_body(settings) -> bool:
    return any(getattr(settings, name) > 0 for name in (*BODY_EFFECTS, "face_small"))


def analyze_body(small_rgb: np.ndarray, width: int, height: int, near=None, analyzer: BodyAnalyzer | None = None,
                 fill=None) -> BodyInfo:
    """줄인 그림(small_rgb)으로 사람 분리 + 몸의 점. 몸의 점은 원래 화면(width×height) 좌표로. near = 보정할 사람 얼굴 자리.

    얼굴 자리를 알면 머리 둘레만 잘라 한 번 더 분리한다: 화면 전체를 256으로 줄이면 4K에서 한 칸이 8~15화소라
    귀걸이·잔머리가 '배경'이 되어 제자리에 남았다 (얼굴 작게에서 두 겹으로 보였다).
    """
    analyzer = analyzer or BodyAnalyzer.shared()
    parts = analyzer.parts(small_rgb)
    sh, sw = small_rgb.shape[:2]
    near_small = None if near is None else (near[0] * sw / width, near[1] * sh / height)
    box = person_box(parts, sw, sh, near_small)
    keypoints = None
    if box is not None:
        keypoints = analyzer.keypoints(small_rgb, box)
        keypoints[:, 0] *= width / sw
        keypoints[:, 1] *= height / sh
    head = None
    if near_small is not None:
        head_box = _head_box(parts, sw, sh, near_small)
        if head_box is not None:
            x0, y0, x1, y1 = head_box
            head = (analyzer.parts(small_rgb[y0:y1, x0:x1]),
                    (round(x0 * width / sw), round(y0 * height / sh), round(x1 * width / sw), round(y1 * height / sh)))
    return BodyInfo(parts, keypoints, head, fill)


def _head_box(parts: np.ndarray, width: int, height: int, near: tuple[float, float]) -> tuple[int, int, int, int] | None:
    """얼굴 자리를 품은 머리(머리카락+얼굴 피부) 덩어리의 상자를 넓힌 것 (width×height 좌표). 너무 작으면 None."""
    head = (parts[..., HAIR].astype(np.float32) + parts[..., FACE_SKIN] > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(head, connectivity=8)
    if count <= 1:
        return None
    nx, ny = int(near[0] / width * PARTS_SIZE), int(near[1] / height * PARTS_SIZE)
    label = labels[min(PARTS_SIZE - 1, max(0, ny)), min(PARTS_SIZE - 1, max(0, nx))]
    if label == 0:
        return None
    x, y, w, h = stats[label, :4]
    sx, sy = width / PARTS_SIZE, height / PARTS_SIZE
    side = max(w * sx, h * sy) * (1 + 2 * HEAD_MARGIN)  # 정사각형 (분리 모델 입력이 정사각형)
    cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
    x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    x1, y1 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    return (x0, y0, x1, y1) if x1 - x0 >= 16 and y1 - y0 >= 16 else None


def body_for_image(rgb: np.ndarray, landmarks: np.ndarray | None, fill=None) -> BodyInfo:
    """샘플 컷·미리보기용: 전체 그림에서 (긴 변 POSE_LONG_SIDE로 줄여) 분석."""
    h, w = rgb.shape[:2]
    scale = min(1.0, POSE_LONG_SIDE / max(w, h))
    small = cv2.resize(rgb, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    near = None if landmarks is None or np.isnan(landmarks).any() else landmarks.mean(axis=0)
    return analyze_body(small, w, h, near, fill=fill)


class BodyFrames:
    """보정본 렌더: 프레임마다 사람 분리 + 몸의 점 (몸의 점은 앞 프레임과 섞어 떨림을 줄인다).

    fill = 이 영상 내내 쓰는 배경 메꾸기 (앞 프레임을 기억하는 services.inpaint.MemoryFill).
    """

    def __init__(self, fill=None) -> None:
        self._previous: np.ndarray | None = None
        self.fill = fill

    def __call__(self, frame, landmarks: np.ndarray | None) -> BodyInfo:
        w, h = frame.width, frame.height
        scale = min(1.0, POSE_LONG_SIDE / max(w, h))
        small = frame.reformat(max(2, round(w * scale)) // 2 * 2, max(2, round(h * scale)) // 2 * 2, format="rgb24").to_ndarray()
        near = None if landmarks is None or np.isnan(landmarks).any() else landmarks.mean(axis=0)
        info = analyze_body(small, w, h, near, fill=self.fill)
        if info.keypoints is not None and self._previous is not None:
            good = (info.keypoints[:, 2] > KEYPOINT_MIN_SCORE) & (self._previous[:, 2] > KEYPOINT_MIN_SCORE)
            mixed = info.keypoints.copy()
            mixed[good, :2] = SMOOTH * info.keypoints[good, :2] + (1 - SMOOTH) * self._previous[good, :2]
            info = BodyInfo(info.parts, mixed, info.head, info.fill)
        self._previous = info.keypoints
        return info


def _point(keypoints: np.ndarray, *indices: int) -> np.ndarray | None:
    rows = keypoints[list(indices)]
    if (rows[:, 2] < KEYPOINT_MIN_SCORE).any():
        return None
    return rows[:, :2].mean(axis=0)


def _edge(alpha: np.ndarray, start: np.ndarray, direction: np.ndarray, limit: float) -> np.ndarray | None:
    """start에서 direction 쪽으로 가며 사람 층이 끝나는 곳 (옷·피부의 실제 가장자리). alpha는 GRID 칸 단위, 좌표는 화소."""
    h, w = alpha.shape
    for distance in np.arange(GRID, limit, GRID):
        x, y = (start + direction * distance) / GRID
        if not (0 <= x < w and 0 <= y < h) or alpha[int(y), int(x)] < 0.5:
            return start + direction * max(0.0, distance - GRID)
    return None


def _arm_near(keypoints: np.ndarray, point: np.ndarray, distance: float) -> bool:
    """팔꿈치·손목이 그 자리 가까이 있나 — 팔이 허리 옆에 붙어 있으면 찾은 가장자리가 허리가 아니라 팔 바깥이다."""
    arm = keypoints[[7, 8, 9, 10]]
    near = np.hypot(arm[:, 0] - point[0], arm[:, 1] - point[1]) < distance
    return bool((near & (arm[:, 2] >= KEYPOINT_MIN_SCORE)).any())


def _pull_disp(xs, ys, pulls) -> tuple[np.ndarray, np.ndarray]:
    """여러 점 당기기를 하나의 매끄러운 변형장으로 (services.beautify._warp와 같은 식). 내용이 옮겨 가는 양."""
    total, mx, my = np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs)
    for (cx, cy), (tx, ty), radius in pulls:  # 반지름 안 칸만 (화면 전체로 하면 당기기 6개에 0.2초)
        r0, r1 = max(0, int((cy - radius) / GRID)), max(0, int((cy + radius) / GRID) + 2)
        c0, c1 = max(0, int((cx - radius) / GRID)), max(0, int((cx + radius) / GRID) + 2)
        box = np.s_[r0:r1, c0:c1]
        weight = np.clip(1 - ((xs[box] - cx) ** 2 + (ys[box] - cy) ** 2) / radius**2, 0, None) ** 2
        total[box] += weight
        mx[box] += weight * (tx - cx)
        my[box] += weight * (ty - cy)
    scale = np.minimum(total, 1.0) / np.maximum(total, 1e-6)
    return mx * scale, my * scale


def _body_pulls(alpha, keypoints, strengths) -> list:
    shoulders, hips = _point(keypoints, 5, 6), _point(keypoints, 11, 12)
    if shoulders is None or hips is None:
        return []
    axis = hips - shoulders
    torso = float(np.linalg.norm(axis))
    if torso < 10:
        return []
    down = axis / torso
    side = np.array([-down[1], down[0]])
    pulls = []
    if strengths.get("waist", 0) > 0:
        for t, share in ((0.62, 0.7), (0.75, 1.0), (0.88, 0.7)):  # 허리선 셋 — 곡선이 매끄럽게
            center = shoulders + axis * t
            for direction in (side, -side):
                edge = _edge(alpha, center, direction, torso)
                if edge is not None and not _arm_near(keypoints, edge, ARM_CLEAR * torso):
                    half = float(np.linalg.norm(edge - center))
                    target = edge - direction * half * WAIST_MAX * share * min(1.0, strengths["waist"])
                    pulls.append((edge, target, 0.45 * torso))
    if strengths.get("shoulders", 0) > 0:
        width = float(np.linalg.norm(keypoints[5, :2] - keypoints[6, :2]))
        for index in (5, 6):
            if keypoints[index, 2] < KEYPOINT_MIN_SCORE:
                continue
            outward = keypoints[index, :2] - shoulders
            length = float(np.linalg.norm(outward))
            if length < 1:
                continue
            outward /= length
            edge = _edge(alpha, keypoints[index, :2], outward, 0.6 * width)
            edge = keypoints[index, :2] if edge is None else edge
            target = edge - outward * width * SHOULDER_MAX * min(1.0, strengths["shoulders"])
            pulls.append((edge, target, 0.35 * torso))
    return pulls


def person_alpha(parts: np.ndarray, width: int, height: int) -> np.ndarray:
    """사람 층의 투명도 (0~1). 확률 0.2 아래는 0으로 — 배경에 깔린 옅은 확률까지 옮기면 배경이 흐릿하게 섞였다."""
    p = np.clip((layer(parts, width, height, PERSON) - 0.2) / 0.6, 0, 1)
    return p * p * (3 - 2 * p)


def _person_layer(body: BodyInfo, cw: int, ch: int) -> np.ndarray:
    """GRID 칸 단위 사람 층: 화면 전체 분리 + 머리 둘레는 다시 분리한 것으로 (가장자리에서 서서히 바꿔 이음매가 없게)."""
    alpha = person_alpha(body.parts, cw, ch)
    if body.head is None:
        return alpha
    head_parts, (x0, y0, x1, y1) = body.head
    c0, r0, c1, r1 = x0 // GRID, y0 // GRID, min(cw, x1 // GRID), min(ch, y1 // GRID)
    if c1 - c0 < 4 or r1 - r0 < 4:
        return alpha
    fine = person_alpha(head_parts, c1 - c0, r1 - r0)
    ramp_y = np.clip(np.minimum(np.arange(r1 - r0), np.arange(r1 - r0)[::-1]) / max(1.0, HEAD_FEATHER * (r1 - r0)), 0, 1)
    ramp_x = np.clip(np.minimum(np.arange(c1 - c0), np.arange(c1 - c0)[::-1]) / max(1.0, HEAD_FEATHER * (c1 - c0)), 0, 1)
    ramp = np.minimum(ramp_y[:, None], ramp_x[None, :]).astype(np.float32)
    alpha[r0:r1, c0:c1] = fine * ramp + alpha[r0:r1, c0:c1] * (1 - ramp)
    return alpha


def fill_telea(image: np.ndarray, box: tuple[int, int, int, int], hole: np.ndarray, need: np.ndarray, carry: np.ndarray) -> np.ndarray:
    """사람 뒤 배경 메꾸기 (기본): 1/8 크기에서 둘레 배경으로 번져 채워(cv2.inpaint) 늘린다. 빠르지만 흐리다.

    box 영역의 메꾼 그림을 돌려준다. hole = 메울 수 있는 곳(사람 자리), need = 실제로 보일 곳(사람이 비켜난 곳),
    carry = 화면 전체의 옮기는 층 투명도 (GRID 칸) — 앞 프레임 배경을 기억하는 메꾸기가 쓴다.
    """
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    small = (max(8, round((x1 - x0) * FILL_SCALE)), max(8, round((y1 - y0) * FILL_SCALE)))
    small_hole = cv2.resize(hole.astype(np.uint8) * 255, small, interpolation=cv2.INTER_NEAREST)
    small_hole = cv2.dilate(small_hole, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(cv2.resize(crop, small, interpolation=cv2.INTER_AREA), small_hole, 3, cv2.INPAINT_TELEA)
    return cv2.resize(filled, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def reshape(image: np.ndarray, landmarks: np.ndarray | None, strengths: dict, body: BodyInfo) -> bool:
    """제자리에서 사람 층만 변형한다 (바뀐 게 있으면 True). strengths = 효과 이름 → 0~1 (face_small은 얼굴 점이 있어야).
    사람 뒤 배경은 body.fill로 메꾼다 (fill_telea와 같은 모양, 없으면 fill_telea).

    변형·투명도·메꿀 자리는 GRID 화소 칸에서 계산하고 (사람 분리가 256 해상도라 더 촘촘해도 얻을 게 없다),
    원래 크기에서는 그림 옮기기(remap)와 섞기만 — 4K에서 화면 전체를 원래 크기로 계산하면 프레임당 1초가 넘었다.
    """
    h, w = image.shape[:2]
    ch, cw = h // GRID + 1, w // GRID + 1  # 칸 (i, j) = 화소 (j·GRID, i·GRID)
    alpha = _person_layer(body, cw, ch)
    ys, xs = np.mgrid[0:ch, 0:cw].astype(np.float32) * GRID
    dx, dy = np.zeros_like(xs), np.zeros_like(xs)
    keypoints = body.keypoints
    if keypoints is not None:
        pulls = _body_pulls(alpha, keypoints, strengths)
        if pulls:
            px, py = _pull_disp(xs, ys, pulls)
            dx, dy = dx + px, dy + py
        if strengths.get("legs", 0) > 0:
            lx, ly = _legs_disp(xs, ys, keypoints, strengths["legs"])
            dx, dy = dx + lx, dy + ly
    if strengths.get("face_small", 0) > 0 and landmarks is not None and not np.isnan(landmarks).any():
        fx, fy = _head_disp(xs, ys, landmarks, strengths["face_small"], body.parts)
        dx, dy = dx + fx, dy + fy
    dx, dy = dx.astype(np.float32), dy.astype(np.float32)  # cv2.remap은 float32만
    # 옮기는 층(carry) = 사람 + 경계 띠 (사람 분리 한 칸 남짓). 사람 분리 경계 바로 밖에도 피부·머리카락 색이 옅게 번져 있어서
    # 사람만 옮기면 그 띠가 제자리에 남아 원래 윤곽이 선으로 보였다 → 띠째 옮기고, 띠째 비켜난 곳만 배경으로 메꾼다
    grow = max(3, round(HOLE_GROW * w / PARTS_SIZE / GRID) * 2 + 1)
    kernel = np.ones((grow, grow), np.uint8)
    carry = np.clip(cv2.GaussianBlur(cv2.dilate(alpha, kernel), (0, 0), grow / 4), 0, 1)
    near_person = (carry > 0.01).astype(np.uint8)
    # 움직임은 사람 둘레에서만 의미가 있다 (배경은 어차피 그대로) — 다리 길게는 엉덩이 아래 화면 폭 전체가 움직여 4K에서 느렸다
    moved = ((np.abs(dx) + np.abs(dy)) > 0.25).astype(np.uint8) & cv2.dilate(near_person, kernel)
    rows, cols = np.nonzero(moved)
    if not len(rows):
        return False
    # 칸에서: 바꿀 곳(region) = 움직인 곳 + 둘레 2칸, 메꿀 곳(hole) = 사람 자리 중 움직인 곳 둘레 1칸 안
    region = cv2.dilate(moved, np.ones((5, 5), np.uint8))
    hole = near_person & cv2.dilate(moved, np.ones((3, 3), np.uint8))
    r0, r1 = max(0, rows.min() - 3), min(ch, rows.max() + 4)
    c0, c1 = max(0, cols.min() - 3), min(cw, cols.max() + 4)
    x0, y0, x1, y1 = c0 * GRID, r0 * GRID, min(w, c1 * GRID), min(h, r1 * GRID)
    size = ((c1 - c0) * GRID, (r1 - r0) * GRID)

    def up(a: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.resize(np.ascontiguousarray(a[r0:r1, c0:c1]), size, interpolation=interpolation)[: y1 - y0, : x1 - x0]

    # 결과 화소는 원래 그림의 (자리 - 옮긴 양)에서 가져온다
    map_x = np.arange(x0, x1, dtype=np.float32)[None, :] - up(dx)
    map_y = np.arange(y0, y1, dtype=np.float32)[:, None] - up(dy)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # 투명도는 칸에서 옮기고 늘린다 (사람 분리가 원래 거칠어서 원래 크기로 옮겨도 같은데, 4K에선 그만큼 느리다)
    moved_carry = cv2.remap(carry, (xs - dx) / GRID, (ys - dy) / GRID, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_alpha, keep_behind = up(moved_carry), up(1 - moved_carry)
    # 메운 배경은 층이 실제로 비켜난 곳에만 — 메울 자리 전체에 깔면 사람 분리가 놓친 옷(흰 소매·밝은 바지)까지
    # 둘레 배경색으로 칠해져 얼룩이 됐다 (가만히 있는 곳은 원래 화소 그대로)
    retreat = cv2.GaussianBlur(np.clip(carry - moved_carry, 0, 1) * hole, (3, 3), 0)
    crop = image[y0:y1, x0:x1]  # 제자리 (image의 일부)
    behind = crop
    if hole[r0:r1, c0:c1].any():  # 사람이 있던 자리의 배경 (사람이 비켜난 곳에만 보인다)
        need = up(retreat)
        filled = (body.fill or fill_telea)(image, (x0, y0, x1, y1), up(hole, cv2.INTER_NEAREST) > 0, need > 0.01, carry)
        behind = cv2.blendLinear(filled, crop, need, 1 - need)
    mixed = cv2.blendLinear(warped, behind, warped_alpha, keep_behind)
    np.copyto(crop, mixed, where=up(region, cv2.INTER_NEAREST)[..., None] > 0)  # 4K에선 crop[mask] = …보다 몇 배 빠르다
    return True


def _legs_disp(xs, ys, keypoints, strength) -> tuple[np.ndarray, np.ndarray]:
    """엉덩이선 아래를 몸통 방향으로 늘린다 (엉덩이선에서 부드럽게 시작)."""
    hips, shoulders = _point(keypoints, 11, 12), _point(keypoints, 5, 6)
    if hips is None or shoulders is None:
        return np.zeros_like(xs), np.zeros_like(xs)
    torso = max(1e-6, float(np.linalg.norm(hips - shoulders)))
    down = (hips - shoulders) / torso
    t = (xs - hips[0]) * down[0] + (ys - hips[1]) * down[1]
    ramp = np.clip(t / (0.25 * torso), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    k = LEGS_MAX * min(1.0, strength)
    move = np.where(t > 0, t * k / (1 + k), 0) * ramp  # 결과의 t 자리는 원래 t/(1+k)에서 → (1+k)배 길게
    return down[0] * move, down[1] * move


def _head_disp(xs, ys, landmarks, strength, parts) -> tuple[np.ndarray, np.ndarray]:
    """머리(얼굴 + 머리카락)를 얼굴 가운데로 고르게 줄이고, 머리 바깥 고리(목·어깨 위)에서 0으로 잦아든다.

    줄이는 범위는 사람 분리의 실제 머리 경계까지 — 얼굴 점 윤곽은 실제 경계보다 안쪽이라 원래 경계가 남았다.
    """
    cx, cy, scale = face_small_scale(landmarks, strength)
    amount = 1.0 - scale
    ch, cw = xs.shape
    head = layer(parts, cw, ch, (HAIR, FACE_SKIN)) > 0.5
    r = np.hypot(xs - cx, ys - cy)
    face_reach = float(np.max(np.hypot(landmarks[:, 0] - cx, landmarks[:, 1] - cy)))
    near = head & (r < HEAD_REACH * face_reach)  # 긴 머리카락이 어깨 아래까지 내려와도 그 끝까지 줄이지는 않게
    reach = max(face_reach, float(r[near].max())) if near.any() else face_reach
    outer = HEAD_FADE * reach
    t = np.clip((outer - r) / (outer - reach), 0, 1)
    shrink = amount * t * t * (3 - 2 * t)  # 머리 안은 고르게 amount, 고리에서 0으로 (접히지 않는다: 바깥으로 갈수록 덜 당김)
    return -(xs - cx) * shrink, -(ys - cy) * shrink


def overlay(image: np.ndarray, body: BodyInfo) -> np.ndarray:
    """적용 범위 그림 (몸매): 사람 층을 칠하고 몸의 점을 잇는다."""
    out = image.copy()
    h, w = out.shape[:2]
    alpha = person_alpha(body.parts, w, h)[..., None]
    out = (out + (np.array((120, 200, 255), np.float32) - out) * alpha * 0.35).astype(np.uint8)
    if body.keypoints is not None:
        thick = max(2, w // 300)
        for a, b in ((5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)):
            if body.keypoints[a, 2] > KEYPOINT_MIN_SCORE and body.keypoints[b, 2] > KEYPOINT_MIN_SCORE:
                cv2.line(out, tuple(map(int, body.keypoints[a, :2])), tuple(map(int, body.keypoints[b, :2])), (255, 220, 0), thick, cv2.LINE_AA)
    return out
