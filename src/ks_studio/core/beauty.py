"""AI 보정 효과의 '어디를 얼마나' 계산. 순수 numpy — Qt·이미지 처리 없음 (화소는 services.beautify가 바꾼다).

랜드마크는 InsightFace 2d106 배치 (화면 기준 왼쪽/오른쪽, 실측으로 확인):
  얼굴 윤곽 0(턱 끝)·1~16(왼쪽)·17~32(오른쪽) · 눈 바깥 끝 35 / 93 · 콧볼 77·78 / 83·84 · 코끝 86 · 콧구멍 아래 가운데 80
  · 입꼬리 52 / 61 · 눈 테두리 8점(가운데 38 / 88, 윗꺼풀 41·40·42 / 95·94·96) · 눈썹 위아래 9점 · 입술 바깥 12점 / 안쪽 8점
  · 콧대 72(눈 사이)·73·74 → 코끝 86 · 코 옆 75·76 / 81·82 · 콧볼 아래 77·78·79·80·85·84·83

효과 (피부·메이크업 색을 먼저 입히고 얼굴형 변형은 마지막에 — 색이 얼굴과 같이 움직이게):
  피부    매끈하게(피부만 흐리게) · 잡티 제거(주변보다 작고 어두운 점을 메꿈) · 톤 밝게 · 다크서클(눈 밑 타원을 밝게)
  얼굴형  얼굴 갸름(볼 아래~턱선을 가운데 선 쪽으로) · 광대 축소(눈 옆 윤곽을 안쪽으로) · 턱 길이(턱 끝을 위/아래로)
          · 눈 키우기(눈 가운데 확대) · 콧볼 축소(콧볼을 코 가운데로) · 코 길이(코 아래쪽을 위로) · 웃는 입(입꼬리를 위로)
          · 얼굴 작게(얼굴 영역만 가운데로 줄이고, 줄어 생긴 틈은 둘레(배경·머리카락·목)로 메꾼다 — 배경은 그대로)
  메이크업 블러셔('볼 사과' 타원에 색) · 윤곽(광대 아래·코 옆은 어둡게, 콧대·이마·턱 끝은 밝게) · 아이섀도(눈두덩에 색)
          · 립 컬러(입술 색만 바꾸고 결은 그대로) · 치아 미백 · 눈 또렷하게(눈 안을 선명하게)
여기서는 '점을 어디로 당길지(Pull)·어디를 키울지(Zoom)·어느 영역(다각형·타원)'만 정한다.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ks_studio.core.jsonio import write_json_atomic

EYE_OUTER = (35, 93)
NOSE_WINGS = ((77, 78), (83, 84))
NOSE_CENTER = (80, 86)
CHEEK_CONTOUR = ((12, 13), (28, 29))  # 왼쪽 1,9~16 ↔ 오른쪽 17,25~32가 같은 순서로 짝 (12↔28, 13↔29)
BLUSH_COLOR = (232, 112, 128)  # RGB, 자연스러운 핑크
NOSE_PULL_MAX = 0.5  # 강도 100%일 때 콧볼이 코 가운데까지 거리의 50%만큼 들어온다
# (2026-11-10: 0.3 → 0.5. 실측 콧볼 폭 −6.3% → −10.6%, CapCut으로 만든 목표 쇼츠가 −10.7%. 콧볼 여러 점을 당겨 봐도 더 낫지 않았다)
NASOLABIAL = (0.04, 0.12, 0.05, 0.09)  # 팔자주름 띠: 콧볼 바깥에서 밖으로 0.04 · 입꼬리에서 밖으로 0.12 · 아래로 0.05 · 반폭 0.09 (눈 사이 거리 대비)
NASOLABIAL_REACH = 1.3  # 띠 길이를 이만큼 늘린다 (타원은 끝으로 갈수록 약해져서, 콧볼·입꼬리 끝까지 지우려면)
NOSE_TIP = 86
CHIN = 0
# 윤곽 점 번호를 위에서 아래 순서로 (왼쪽 1,9~16,2~8 / 오른쪽 17,25~32,18~24, 같은 자리끼리 짝). 볼 아래~턱선 구간을 당긴다
JAW_LEFT = (13, 14, 15, 16, 2, 3, 4)
JAW_RIGHT = (29, 30, 31, 32, 18, 19, 20)
SLIM_MAX = 0.06  # 강도 100%일 때 턱선이 눈 사이 거리의 6%만큼 들어온다
SLIM_RADIUS = 0.35  # 당기는 범위 (눈 사이 거리 대비)
FRONTAL_FULL, FRONTAL_NONE = 0.15, 0.35  # 코끝이 두 눈 가운데에서 눈 사이 거리의 15%까지 벗어나면 100%, 35%면 0
FACE_SIZE_FULL, FACE_SIZE_NONE = 0.03, 0.015  # 눈 사이 거리가 화면 폭의 3% 이상이면 100%, 1.5% 이하면 0 (뒤쪽 작은 얼굴)
CONTOUR = tuple(range(33))
EYES = ((35, 41, 40, 42, 39, 37, 33, 36), (89, 95, 94, 96, 93, 91, 87, 90))  # 눈 테두리 (화면 왼쪽 눈, 오른쪽 눈)
EYE_CENTERS = (38, 88)
EYE_CORNERS = ((35, 39), (93, 89))  # (바깥, 안쪽)
BROWS = ((43, 44, 45, 47, 46, 48, 49, 51, 50), (97, 98, 99, 100, 101, 102, 103, 104, 105))
BROW_TOPS = (48, 49, 51, 50, 102, 103, 104, 105)
LIPS_OUTER = (52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55)
LIPS_INNER = (65, 66, 62, 70, 69, 57, 60, 54)
CHEEKBONE_LEFT, CHEEKBONE_RIGHT = (10, 11, 12), (26, 27, 28)  # 눈 바깥 끝 옆~아래 윤곽
CHIN_POINTS = (0, 8, 24, 7, 23)  # 턱 끝과 양옆
FOREHEAD_LIFT = 0.25  # 피부 영역을 눈썹 위로 눈 사이 거리의 25%만큼 (이마 아래쪽; 머리카락선은 모른다)
CHEEKBONE_MAX, CHEEKBONE_RADIUS = 0.05, 0.3
CHIN_MAX, CHIN_RADIUS = 0.08, 0.35  # 강도 ±100%일 때 턱 끝이 눈 사이 거리의 8%만큼 아래(+)/위(-)로
NOSE_LENGTH_MAX = 0.12  # 강도 100%일 때 코 아래쪽이 (두 눈 가운데~코 아래) 거리의 12%만큼 올라간다
FACE_SMALL_MAX = 0.08  # 강도 100%일 때 머리(얼굴+머리카락)가 8% 작아진다 (드러난 배경을 메우는 폭도 그만큼이라 너무 키우면 메운 티가 난다)
EYE_ZOOM_MAX, EYE_ZOOM_RADIUS = 0.15, 1.0  # 강도 100%일 때 눈 가운데가 1/(1-0.15)≈1.18배 · 범위 = 눈 가로 길이 × 1.0
UNDER_EYE = (0.45, 0.55, 0.22)  # 다크서클 타원: 눈 가운데에서 아래로 눈 가로 × 0.45, 반지름 가로 0.55 · 세로 0.22
NOSE_BOTTOM = (76, 77, 78, 79, 80, 85, 84, 83, 82, 86)  # 코 옆 아래~콧볼·콧구멍 (잡티 제거에서 뺀다 — 콧구멍·콧볼 그늘도 '어두운 점')
NOSE_BRIDGE = (72, 86)  # 콧대 하이라이트: 눈 사이 → 코끝
NOSE_SIDES = ((75, 77), (81, 83))  # 코 옆 쉐딩: 코 옆 위 → 콧볼
MOUTH_CORNERS = (52, 61)
SMILE_MAX, SMILE_RADIUS = 0.04, 0.22  # 강도 100%일 때 입꼬리가 눈 사이 거리의 4%만큼 위로 (바깥으로도 조금)
# 눈두덩 = 윗꺼풀 점에서 같은 자리 눈썹 아랫점 쪽으로 EYESHADOW_HEIGHT만큼 올린 띠 (바깥 끝 → 안쪽 끝 순서로 짝)
EYELIDS = ((35, 41, 40, 42, 39), (93, 96, 94, 95, 89))
BROW_LOWS = ((43, 44, 45, 47, 46), (101, 100, 99, 98, 97))
EYESHADOW_HEIGHT = 0.55
EYESHADOW_COLORS = {  # 칸 값 → (이름, RGB)
    "brown": ("브라운", (150, 100, 80)),
    "pink": ("핑크", (220, 130, 150)),
    "coral": ("코랄", (235, 125, 95)),
    "peach": ("피치", (240, 170, 130)),
    "gold": ("골드", (205, 165, 95)),
    "burgundy": ("버건디", (140, 60, 70)),
    "purple": ("퍼플", (150, 105, 175)),
    "smoky": ("스모키", (105, 95, 95)),
}
LIP_COLORS = {  # 칸 값 → (이름, RGB)
    "rose": ("로즈", (196, 64, 92)),
    "coral": ("코랄", (232, 106, 90)),
    "red": ("레드", (190, 30, 45)),
    "pink": ("핑크", (232, 112, 152)),
    "nude": ("누드", (198, 128, 112)),
    "plum": ("플럼", (138, 50, 80)),
    "orange": ("오렌지", (236, 112, 52)),
    "mlbb": ("MLBB", (176, 92, 92)),
}
LIP_GLOSS_SHEEN = (0.55, 0.42, 0.30)  # 광택 하이라이트 타원: 아랫입술에서 안쪽 쪽으로 0.55 · 반지름 가로(입 폭 대비) 0.42 · 세로(아랫입술 두께 대비) 0.30
EFFECTS = (  # BeautySettings의 효과 세기 칸 (chin만 -1~1, 나머지 0~1)
    "smooth", "blemish", "brighten", "dark_circle", "nasolabial",
    "slim", "cheekbone", "chin", "eyes", "nose_slim", "nose_length", "smile", "face_small",
    "blush", "contour", "eyeshadow", "lip", "lip_gloss", "teeth", "eye_clear",
    "waist", "shoulders", "legs", "bg_blur",
)
PRESET_FIELDS = (*EFFECTS, "eyeshadow_color", "lip_color")  # 프리셋에 담는 칸 (효과 세기 + 색)


@dataclass(frozen=True)
class BeautySettings:
    smooth: float = 0.0  # 피부 매끈하게 0~1
    blemish: float = 0.0  # 잡티 제거
    brighten: float = 0.0  # 피부 톤 밝게
    dark_circle: float = 0.0  # 다크서클 지우기
    nasolabial: float = 0.0  # 팔자주름 지우기 (100% = 그늘을 거의 다)
    slim: float = 0.0  # 얼굴 갸름
    cheekbone: float = 0.0  # 광대 축소
    chin: float = 0.0  # 턱 길이 -1(짧게)~1(길게)
    eyes: float = 0.0  # 눈 키우기
    nose_slim: float = 0.0  # 콧볼 축소
    nose_length: float = 0.0  # 코 길이 줄이기
    smile: float = 0.0  # 웃는 입
    face_small: float = 0.0  # 얼굴 작게
    blush: float = 0.0  # 블러셔
    contour: float = 0.0  # 윤곽 메이크업 (쉐딩·하이라이트)
    eyeshadow: float = 0.0  # 아이섀도
    eyeshadow_color: str = "brown"  # EYESHADOW_COLORS의 키
    lip: float = 0.0  # 립 컬러
    lip_color: str = "rose"  # LIP_COLORS의 키
    lip_gloss: float = 0.0  # 립 광택 (촉촉하게 반짝이는 입술)
    teeth: float = 0.0  # 치아 미백
    eye_clear: float = 0.0  # 눈 또렷하게
    waist: float = 0.0  # 허리 잘록 (몸매 — 몸의 점·사람 분리 모델)
    shoulders: float = 0.0  # 어깨 좁게
    legs: float = 0.0  # 다리 길게
    bg_blur: float = 0.0  # 배경 흐림 (인물 모드 — 사람 분리 모델로 보정할 사람 말고 다 흐리게)
    fill: str = "migan"  # 몸매·얼굴 작게로 드러난 배경 메꾸기: telea(빠르게) · migan · lama (services.inpaint.FILLS)
    approved: bool = False  # 샘플 컷을 보고 OK했는지 — 그 뒤에야 미리보기와 보정본 만들기를 한다
    target: tuple[float, ...] | None = None  # 보정할 사람의 얼굴 인식값 (없으면 가장 큰 얼굴)
    blur_others: bool = False  # 주인공·흐리지 않을 사람 말고 다른 얼굴은 모두 흐리게 (지나가는 사람 보호)
    blur_style: str = "mosaic"  # core.face_blur.BLUR_STYLES의 키
    keep: tuple[tuple[float, ...], ...] = ()  # 이 영상에서 흐리지 않을 사람들의 인식값 (등록한 사람은 앱 전체에서 따로)

    @property
    def is_identity(self) -> bool:
        """보정 효과가 하나도 없다 (얼굴 흐리기는 따로 — needs_render)."""
        return all(getattr(self, name) == 0 for name in EFFECTS)

    @property
    def needs_render(self) -> bool:
        """보정본을 만들 일이 있다: 효과가 있거나 다른 얼굴을 흐린다."""
        return not self.is_identity or self.blur_others

    @property
    def keep_vectors(self) -> list[np.ndarray]:
        return [np.array(k, np.float32) for k in self.keep]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def target_vector(self) -> np.ndarray | None:
        return None if self.target is None else np.array(self.target, np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "BeautySettings":
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if values.get("target") is not None:
            values["target"] = tuple(values["target"])
        values["keep"] = tuple(tuple(float(x) for x in k) for k in values.get("keep", ()) if len(k) == 512)
        return cls(**values)


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float  # 라디안 (얼굴 기울기)


@dataclass(frozen=True)
class Pull:
    """(cx, cy) 근처 화소를 (tx, ty) 쪽으로 당긴다. radius 밖은 그대로."""

    cx: float
    cy: float
    tx: float
    ty: float
    radius: float


def eye_distance(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[EYE_OUTER[1]] - lm[EYE_OUTER[0]]))


def face_angle(lm: np.ndarray) -> float:
    dx, dy = lm[EYE_OUTER[1]] - lm[EYE_OUTER[0]]
    return math.atan2(dy, dx)


def frontal_weight(lm: np.ndarray) -> float:
    """얼굴이 정면일수록 1, 옆으로 돌수록 0. 옆얼굴에 블러셔를 칠하면 머리카락·귀에 묻는다 (실측) →
    효과를 이만큼만 입힌다. 0/1로 끊지 않고 서서히 바꿔서 고개를 돌릴 때 깜빡이지 않게."""
    eyes = lm[EYE_OUTER[1]] - lm[EYE_OUTER[0]]
    distance = float(np.linalg.norm(eyes))
    if distance < 1e-6:
        return 0.0
    middle = (lm[EYE_OUTER[0]] + lm[EYE_OUTER[1]]) / 2
    offset = abs(float(np.dot(lm[NOSE_TIP] - middle, eyes / distance))) / distance
    return float(np.clip((FRONTAL_NONE - offset) / (FRONTAL_NONE - FRONTAL_FULL), 0.0, 1.0))


def size_weight(lm: np.ndarray, frame_width: int) -> float:
    """화면에서 얼굴이 충분히 크면 1, 뒤쪽 행인처럼 작으면 0."""
    ratio = eye_distance(lm) / frame_width
    return float(np.clip((ratio - FACE_SIZE_NONE) / (FACE_SIZE_FULL - FACE_SIZE_NONE), 0.0, 1.0))


def cheek_ellipses(lm: np.ndarray) -> list[Ellipse]:
    """양쪽 볼 사과 자리. 크기는 눈 사이 거리에 비례 (얼굴이 멀어도 같은 비율)."""
    eyes, angle = eye_distance(lm), face_angle(lm)
    out = []
    for side in (0, 1):
        contour = lm[list(CHEEK_CONTOUR[side])].mean(axis=0)
        wing = lm[NOSE_WINGS[side][0]]
        eye = lm[EYE_OUTER[side]]
        cx, cy = 0.45 * contour + 0.25 * wing + 0.30 * eye
        out.append(Ellipse(float(cx), float(cy), 0.22 * eyes, 0.15 * eyes, angle))
    return out


def nose_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """양쪽 콧볼(바깥·아래 점의 가운데)을 코 가운데로 strength만큼 당긴다."""
    if strength <= 0:
        return []
    center = lm[list(NOSE_CENTER)].mean(axis=0)
    width = float(np.linalg.norm(lm[NOSE_WINGS[1][0]] - lm[NOSE_WINGS[0][0]]))
    pulls = []
    for outer, lower in NOSE_WINGS:
        wing = (lm[outer] + lm[lower]) / 2
        target = wing + (center - wing) * NOSE_PULL_MAX * min(1.0, strength)
        pulls.append(Pull(float(wing[0]), float(wing[1]), float(target[0]), float(target[1]), 0.55 * width))
    return pulls


def _face_axis(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """얼굴 가운데 선: (두 눈 가운데, 턱 끝 쪽 단위 벡터). 고개가 기울어도 따라간다."""
    top = (lm[EYE_OUTER[0]] + lm[EYE_OUTER[1]]) / 2
    axis = lm[CHIN] - top
    return top, axis / (np.linalg.norm(axis) + 1e-9)


def _inward_pulls(lm: np.ndarray, indices, amount: float, radius: float) -> list[Pull]:
    """윤곽 점들을 같은 높이의 가운데 선 위 점 쪽으로 amount(px)만큼."""
    top, axis = _face_axis(lm)
    pulls = []
    for index in indices:
        point = lm[index]
        inward = top + axis * np.dot(point - top, axis) - point
        length = float(np.linalg.norm(inward))
        if length < 1e-6:
            continue
        target = point + inward / length * amount
        pulls.append(Pull(float(point[0]), float(point[1]), float(target[0]), float(target[1]), radius))
    return pulls


def slim_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """볼 아래~턱선 점들을 얼굴 가운데 선 쪽으로."""
    if strength <= 0:
        return []
    eyes = eye_distance(lm)
    return _inward_pulls(lm, (*JAW_LEFT, *JAW_RIGHT), SLIM_MAX * eyes * min(1.0, strength), SLIM_RADIUS * eyes)


def cheekbone_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """눈 옆~아래 윤곽(광대)을 안쪽으로."""
    if strength <= 0:
        return []
    eyes = eye_distance(lm)
    indices = (*CHEEKBONE_LEFT, *CHEEKBONE_RIGHT)
    return _inward_pulls(lm, indices, CHEEKBONE_MAX * eyes * min(1.0, strength), CHEEKBONE_RADIUS * eyes)


def chin_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """턱 끝과 양옆을 얼굴 가운데 선 방향으로: strength > 0이면 아래(길게), < 0이면 위(짧게)."""
    if strength == 0:
        return []
    eyes = eye_distance(lm)
    _, axis = _face_axis(lm)
    move = axis * CHIN_MAX * eyes * float(np.clip(strength, -1.0, 1.0))
    return [
        Pull(float(lm[i][0]), float(lm[i][1]), float(lm[i][0] + move[0]), float(lm[i][1] + move[1]), CHIN_RADIUS * eyes)
        for i in CHIN_POINTS
    ]


def nose_length_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """코 아래쪽(코끝~콧구멍 아래 가운데)을 두 눈 가운데 쪽으로 올린다."""
    if strength <= 0:
        return []
    top, _ = _face_axis(lm)
    bottom = lm[list(NOSE_CENTER)].mean(axis=0)
    width = float(np.linalg.norm(lm[NOSE_WINGS[1][0]] - lm[NOSE_WINGS[0][0]]))
    target = bottom + (top - bottom) * NOSE_LENGTH_MAX * min(1.0, strength)
    return [Pull(float(bottom[0]), float(bottom[1]), float(target[0]), float(target[1]), 0.9 * width)]


@dataclass(frozen=True)
class Zoom:
    """(cx, cy) 둘레 radius 안을 확대. 가운데가 1/(1-amount)배, 가장자리로 갈수록 1배."""

    cx: float
    cy: float
    radius: float
    amount: float


def face_outline(lm: np.ndarray) -> np.ndarray:
    """얼굴 작게가 줄이는 영역: 윤곽 + 눈썹 (이마 위·머리카락은 넣지 않는다 — 머리카락을 가로질러 자르면 선이 보였다)."""
    return hull(np.concatenate([lm[list(CONTOUR)], *(lm[list(b)] for b in BROWS)]))


def face_small_scale(lm: np.ndarray, strength: float) -> tuple[float, float, float] | None:
    """얼굴 작게: (가운데 x, 가운데 y, 배율). 가운데 = 두 눈 가운데와 턱 끝 사이(눈 쪽 40%) — 얼굴이 고르게 줄어든다."""
    if strength <= 0:
        return None
    top, _ = _face_axis(lm)
    center = top + (lm[CHIN] - top) * 0.4
    return float(center[0]), float(center[1]), 1.0 - FACE_SMALL_MAX * min(1.0, strength)


def eye_width(lm: np.ndarray, side: int) -> float:
    outer, inner = EYE_CORNERS[side]
    return float(np.linalg.norm(lm[outer] - lm[inner]))


def eye_zooms(lm: np.ndarray, strength: float) -> list[Zoom]:
    if strength <= 0:
        return []
    return [
        Zoom(float(lm[c][0]), float(lm[c][1]), EYE_ZOOM_RADIUS * eye_width(lm, side), EYE_ZOOM_MAX * min(1.0, strength))
        for side, c in enumerate(EYE_CENTERS)
    ]


def hull(points: np.ndarray) -> np.ndarray:
    """볼록 껍질 (Andrew monotone chain)."""
    pts = sorted(map(tuple, np.asarray(points, float).tolist()))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def skin_polygon(lm: np.ndarray) -> np.ndarray:
    """피부 영역의 바깥 경계: 얼굴 윤곽 + 눈썹 윗점을 이마 쪽으로 올린 점들의 볼록 껍질 (눈·눈썹·입은 따로 뺀다)."""
    _, axis = _face_axis(lm)
    lifted = lm[list(BROW_TOPS)] - axis * FOREHEAD_LIFT * eye_distance(lm)
    return hull(np.concatenate([lm[list(CONTOUR)], lifted]))


def eye_polygons(lm: np.ndarray) -> list[np.ndarray]:
    return [lm[list(eye)] for eye in EYES]


def brow_polygons(lm: np.ndarray) -> list[np.ndarray]:
    return [hull(lm[list(brow)]) for brow in BROWS]


def lip_polygons(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(입술 바깥, 입 안쪽). 입술 = 바깥 - 안쪽, 치아는 안쪽에서만 찾는다."""
    return lm[list(LIPS_OUTER)], lm[list(LIPS_INNER)]


def gloss_ellipse(lm: np.ndarray) -> Ellipse:
    """립 광택 하이라이트 자리: 아랫입술 가운데, 입 안쪽 쪽으로 치우친 납작한 타원 (빛이 닿아 반짝이는 곳)."""
    outer, inner = lip_polygons(lm)
    corner_l, corner_r = lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]]
    mid_x = (corner_l[0] + corner_r[0]) / 2
    bottom = outer[np.argmax(outer[:, 1])]  # 아랫입술 바깥 끝
    inner_bottom = inner[np.argmin(np.abs(inner[:, 0] - mid_x) + 1e6 * (inner[:, 1] < (corner_l[1] + corner_r[1]) / 2))]
    toward, rx_share, ry_share = LIP_GLOSS_SHEEN
    center = bottom + (inner_bottom - bottom) * toward
    width = float(np.linalg.norm(corner_r - corner_l))
    thickness = max(1.0, float(np.linalg.norm(bottom - inner_bottom)))
    angle = math.atan2(corner_r[1] - corner_l[1], corner_r[0] - corner_l[0])
    return Ellipse(float(center[0]), float(center[1]), width * rx_share / 2, thickness * ry_share, angle)


def nostril_polygon(lm: np.ndarray) -> np.ndarray:
    return hull(lm[list(NOSE_BOTTOM)])


def smile_pulls(lm: np.ndarray, strength: float) -> list[Pull]:
    """입꼬리를 위로(얼굴 가운데 선 반대 방향), 바깥으로 조금."""
    if strength <= 0:
        return []
    eyes = eye_distance(lm)
    _, axis = _face_axis(lm)
    middle = lm[list(MOUTH_CORNERS)].mean(axis=0)
    pulls = []
    for corner in MOUTH_CORNERS:
        point = lm[corner]
        outward = (point - middle) / (np.linalg.norm(point - middle) + 1e-9)
        target = point + (-axis + 0.3 * outward) * SMILE_MAX * eyes * min(1.0, strength)
        pulls.append(Pull(float(point[0]), float(point[1]), float(target[0]), float(target[1]), SMILE_RADIUS * eyes))
    return pulls


def _ellipse_along(a: np.ndarray, b: np.ndarray, half_width: float) -> Ellipse:
    """a~b를 긴 축으로 하는 가는 타원."""
    (cx, cy), (dx, dy) = (a + b) / 2, b - a
    return Ellipse(float(cx), float(cy), float(np.hypot(dx, dy)) / 2, half_width, math.atan2(dy, dx))


def contour_shapes(lm: np.ndarray) -> tuple[list[Ellipse], list[Ellipse]]:
    """윤곽 메이크업 자리 (쉐딩, 하이라이트).

    쉐딩: 광대 아래(볼 윤곽 → 입꼬리 쪽으로 비스듬히) · 코 옆. 하이라이트: 콧대 · 이마 가운데 · 턱 끝 안쪽.
    """
    eyes = eye_distance(lm)
    _, axis = _face_axis(lm)
    shade = []
    for contour, corner in ((CHEEK_CONTOUR[0][1], MOUTH_CORNERS[0]), (CHEEK_CONTOUR[1][1], MOUTH_CORNERS[1])):
        start = lm[contour]
        shade.append(_ellipse_along(start, start + (lm[corner] - start) * 0.45, 0.07 * eyes))
    for top, wing in NOSE_SIDES:
        shade.append(_ellipse_along(lm[top], lm[wing], 0.04 * eyes))
    brow_middle = (lm[BROWS[0][4]] + lm[BROWS[1][4]]) / 2  # 두 눈썹 안쪽 끝 가운데
    forehead = brow_middle - axis * 0.22 * eyes
    chin = lm[CHIN] - axis * 0.1 * eyes
    angle = face_angle(lm)
    highlight = [
        _ellipse_along(lm[NOSE_BRIDGE[0]], lm[NOSE_BRIDGE[1]], 0.035 * eyes),
        Ellipse(float(forehead[0]), float(forehead[1]), 0.16 * eyes, 0.09 * eyes, angle),
        Ellipse(float(chin[0]), float(chin[1]), 0.1 * eyes, 0.06 * eyes, angle),
    ]
    return shade, highlight


def eyelid_polygons(lm: np.ndarray) -> list[np.ndarray]:
    """양쪽 눈두덩: 윗꺼풀 선 + 눈썹 쪽으로 올린 선."""
    out = []
    for lid, brow in zip(EYELIDS, BROW_LOWS):
        lower = lm[list(lid)]
        upper = lower + (lm[list(brow)] - lower) * EYESHADOW_HEIGHT
        out.append(np.concatenate([lower, upper[::-1]]))
    return out


def under_eye_ellipses(lm: np.ndarray) -> list[Ellipse]:
    """양쪽 눈 밑 다크서클 자리."""
    _, axis = _face_axis(lm)
    angle = face_angle(lm)
    down, rx, ry = UNDER_EYE
    out = []
    for side, c in enumerate(EYE_CENTERS):
        width = eye_width(lm, side)
        cx, cy = lm[c] + axis * down * width
        out.append(Ellipse(float(cx), float(cy), rx * width, ry * width, angle))
    return out


def nasolabial_ellipses(lm: np.ndarray) -> list[Ellipse]:
    """양쪽 팔자주름 자리: 콧볼 바깥 조금 밖 → 입꼬리 바깥 조금 아래로 이어지는 가는 띠."""
    eyes = eye_distance(lm)
    top, axis = _face_axis(lm)
    middle = lm[list(NOSE_CENTER)].mean(axis=0)
    out_wing, out_mouth, down, half = NASOLABIAL
    shapes = []
    for (wing, _), corner in zip(NOSE_WINGS, MOUTH_CORNERS):
        side = lm[wing] - middle
        side = side - axis * float(np.dot(side, axis))  # 얼굴 가로 방향 (바깥쪽)
        side /= np.linalg.norm(side) + 1e-9
        start = lm[wing] + side * out_wing * eyes
        end = lm[corner] + side * out_mouth * eyes + axis * down * eyes
        shape = _ellipse_along(start, end, half * eyes)
        shapes.append(Ellipse(shape.cx, shape.cy, shape.rx * NASOLABIAL_REACH, shape.ry, shape.angle))
    return shapes


def smooth_track(landmarks: np.ndarray, radius: int = 2) -> np.ndarray:
    """프레임마다 랜드마크의 떨림을 앞뒤 radius 프레임 평균으로 줄인다 (영상 전체를 미리 분석하므로 앞뒤를 다 볼 수 있다).

    landmarks: (프레임 수, 106, 2), 얼굴이 없는 프레임은 NaN. 얼굴이 없던 프레임은 radius 안에 얼굴이 있으면 채운다
    (한두 프레임 놓쳐도 효과가 깜빡이지 않게).
    """
    present = ~np.isnan(landmarks[:, 0, 0])
    filled = np.where(present[:, None, None], landmarks, 0.0)
    out = np.full_like(landmarks, np.nan)
    n = len(landmarks)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        count = present[lo:hi].sum()
        if count:
            out[i] = filled[lo:hi].sum(axis=0) / count
    return out


@dataclass(frozen=True)
class AiState:
    """AI 보정 탭의 저장 내용 (<프로젝트>/ai/settings.json). 컷편집의 project.json과 따로 둔다 (탭이 서로 독립)."""

    settings: dict  # 원본(source 문자열) → BeautySettings
    imported: tuple[str, ...] = ()  # AI 탭에서 직접 가져온 원본
    preview: bool = True  # 미리보기 켜기 (느린 PC는 끈다)
    memos: dict = field(default_factory=dict)  # 원본 → 메모 (목록에서 바로 적는다)

    def for_source(self, source: str) -> BeautySettings:
        return self.settings.get(source, BeautySettings())


def load_ai_state(path: Path) -> AiState:
    if not path.exists():
        return AiState({})
    data = json.loads(path.read_text(encoding="utf-8"))
    return AiState(
        {k: BeautySettings.from_dict(v) for k, v in data.get("settings", {}).items()},
        tuple(data.get("imported", [])),
        bool(data.get("preview", True)),
        {k: str(v) for k, v in data.get("memos", {}).items()},
    )


def save_ai_state(path: Path, state: AiState) -> None:
    data = {
        "settings": {k: v.to_dict() for k, v in state.settings.items()},
        "imported": list(state.imported),
        "preview": state.preview,
        "memos": {k: v for k, v in state.memos.items() if v},
    }
    write_json_atomic(path, data)
