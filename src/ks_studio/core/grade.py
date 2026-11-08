"""색 보정 (원본 영상마다): 상용 편집 프로그램의 '기본 보정'처럼 빛 · 색 · 색상별(HSL) · LUT. 순수 파이썬 + numpy.

보정값은 3D LUT(33×33×33 색 변환표) 한 장으로 굽는다 (build_lut). 영상에는 그 표를 FFmpeg lut3d로 입히고
(services.grade — C로 돌아 4K도 빠르다), 화면 미리보기는 같은 표를 numpy로 입힌다 (sample_lut). 그래서 미리보기와 결과물이 같다.

색은 원본 → (빛·화이트 밸런스는 선형 빛에서) → 밝기 곡선 → 바랜 느낌 → 채도 → 색상별 → 외부 LUT 순서로 바꾼다.
값은 모두 -1~1 (바랜 느낌·LUT 강도만 0~1). 화면에서는 -100~100 %로 보인다.
"""

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from ks_studio.core.jsonio import write_json_atomic

LUT_SIZE = 33
GAMMA = 2.2  # 화면 색 ↔ 선형 빛 (노출·화이트 밸런스는 선형 빛에서 해야 자연스럽다)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # BT.709 밝기
EXPOSURE_STOPS = 2.0  # 노출 100% = 2스톱 (4배 밝게)
TEMPERATURE_GAIN = 0.3  # 색온도 100%: 빨강 ×1.3 · 파랑 ×0.7 (밝기는 그대로)
TINT_GAIN = 0.25  # 색조 100%: 초록 ×0.75 (+ = 자홍 쪽)
HUE_SHIFT_MAX = 30.0  # 색상별 '색상' 100% = 색상환에서 30° 돌리기

LIGHT = {  # 칸 → 이름
    "exposure": "노출", "contrast": "대비", "highlights": "밝은 부분", "shadows": "어두운 부분",
    "whites": "흰색", "blacks": "검정",
}
COLOR = {"temperature": "색온도 (− 차갑게 · + 따뜻하게)", "tint": "색조 (− 초록 · + 자홍)", "vibrance": "생동감", "saturation": "채도"}
HSL_COLORS = {  # 키 → (이름, 색상환 각도) — Lightroom 'HSL'과 같은 8가지
    "red": ("빨강", 0.0), "orange": ("주황", 30.0), "yellow": ("노랑", 60.0), "green": ("초록", 120.0),
    "aqua": ("하늘", 180.0), "blue": ("파랑", 225.0), "purple": ("보라", 270.0), "magenta": ("자홍", 315.0),
}
NO_HSL = ((0.0, 0.0, 0.0),) * len(HSL_COLORS)


@dataclass(frozen=True)
class ColorGrade:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    vibrance: float = 0.0  # 덜 진한 색을 더 많이 (피부는 덜 바뀐다)
    saturation: float = 0.0
    fade: float = 0.0  # 바랜 느낌 0~1 (검정이 뜨고 흰색이 가라앉는다 — 필름 느낌)
    hsl: tuple[tuple[float, float, float], ...] = NO_HSL  # HSL_COLORS 순서로 (색상, 채도, 밝기)
    lut: str = ""  # 외부 .cube 파일 (Workspace.to_source 형식, 빈 값 = 없음)
    lut_strength: float = 1.0

    @property
    def is_identity(self) -> bool:
        """아무것도 바꾸지 않는다 (LUT는 강도가 0이면 없는 것과 같다)."""
        return replace(self, lut="", lut_strength=1.0) == ColorGrade() and (not self.lut or self.lut_strength <= 0)

    def to_dict(self) -> dict:
        return {k: (list(map(list, v)) if k == "hsl" else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ColorGrade":
        """모르는 칸은 버리고, 타입이 맞지 않는 값은 기본값으로 (손으로 고친 파일·예전 버전)."""
        values = {}
        for f in fields(cls):
            value = data.get(f.name)
            if f.name == "hsl":
                try:
                    rows = tuple(tuple(float(v) for v in row[:3]) for row in value)
                    if len(rows) == len(HSL_COLORS) and all(len(r) == 3 for r in rows):
                        values["hsl"] = rows
                except (TypeError, ValueError):
                    pass
            elif f.name == "lut":
                if isinstance(value, str):
                    values["lut"] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[f.name] = float(value)
        return cls(**values)

    def look(self) -> "ColorGrade":
        """프리셋으로 저장·불러올 부분 (외부 LUT 파일은 프로젝트마다 달라서 뺀다)."""
        return replace(self, lut="", lut_strength=1.0)


BUILTIN_LOOKS = {  # 기본 제공 모양 (프리셋 목록 맨 위)
    "따뜻하게": ColorGrade(temperature=0.35, vibrance=0.15),
    "시원하게": ColorGrade(temperature=-0.35, tint=-0.05, contrast=0.05),
    "선명하게": ColorGrade(contrast=0.3, vibrance=0.35, shadows=0.1),
    "화사하게 (밝은 피부)": ColorGrade(exposure=0.08, shadows=0.25, highlights=-0.15, vibrance=0.1, temperature=0.08),
    "부드럽게": ColorGrade(contrast=-0.2, highlights=-0.2, shadows=0.2, fade=0.25),
    "필름": ColorGrade(contrast=-0.1, fade=0.45, saturation=-0.15, temperature=0.12),
    "영화처럼 (틸·오렌지)": ColorGrade(contrast=0.2, highlights=-0.2, saturation=-0.05, hsl=(
        (0.0, 0.1, 0.0), (0.0, 0.25, 0.05), (-0.3, -0.2, 0.0), (0.4, -0.4, 0.0),
        (0.2, 0.25, -0.1), (-0.2, 0.2, -0.1), (0.0, -0.3, 0.0), (0.0, -0.3, 0.0))),
    "흑백": ColorGrade(saturation=-1.0, contrast=0.2),
}


# ── 계산 ────────────────────────────────────────────────────────


def _smooth(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _push(L: np.ndarray, amount: float, weight: np.ndarray, strength: float) -> np.ndarray:
    """L을 weight만큼 밝게(+)·어둡게(-). 끝(0·1)을 넘지 않게 남은 거리에 비례해서."""
    if amount == 0:
        return L
    room = (1 - L) if amount > 0 else L
    return L + strength * amount * weight * room


def tone(L: np.ndarray, g: ColorGrade) -> np.ndarray:
    """밝기 곡선: 밝은 부분·어두운 부분·흰색·검정·대비 (밝기 0~1)."""
    L = np.clip(L, 0.0, 1.0)
    L = _push(L, g.highlights, _smooth(0.35, 1.0, L), 0.35)
    L = _push(L, g.shadows, 1 - _smooth(0.0, 0.6, L), 0.35)
    L = _push(L, g.whites, _smooth(0.65, 1.0, L), 0.3)
    L = _push(L, g.blacks, 1 - _smooth(0.0, 0.35, L), 0.3)
    if g.contrast > 0:  # S자 곡선 쪽으로
        L = L + g.contrast * (L * L * (3 - 2 * L) - L)
    elif g.contrast < 0:  # 가운데(0.5) 쪽으로 모은다
        L = L + g.contrast * 0.5 * (L - 0.5)
    return L


def _hue(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(색상 0~360°, 채도 0~1, 명도 0~1) — HSV."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta > 1e-6, delta, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    h = np.where(delta > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, delta / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return h, s, mx


def _rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 360.0) / 60.0
    c = v * s
    x = c * (1 - np.abs(h % 2 - 1))
    m = v - c
    zero = np.zeros_like(h)
    sector = np.floor(h).astype(np.int32) % 6
    r = np.choose(sector, [c, x, zero, zero, x, c])
    g = np.choose(sector, [x, c, c, x, zero, zero])
    b = np.choose(sector, [zero, zero, x, c, c, x])
    return np.stack([r + m, g + m, b + m], axis=-1)


def _around_wheel(h: np.ndarray, values: list[float]) -> np.ndarray:
    """색상(°)마다 8가지 색 값 사이를 이어 붙인 값 — 이웃한 두 색 사이에서 고르게 바뀐다 (색이 갑자기 끊기지 않게)."""
    centers = [angle for _name, angle in HSL_COLORS.values()]
    xs = [centers[-1] - 360.0, *centers, centers[0] + 360.0]
    ys = [values[-1], *values, values[0]]
    return np.interp(h, xs, ys)


def hsl_adjust(rgb: np.ndarray, hsl: tuple) -> np.ndarray:
    """색상별 색상·채도·밝기. 회색·아주 어두운 곳은 색이 정해지지 않아 거의 그대로."""
    if all(v == 0 for row in hsl for v in row):
        return rgb
    h, s, v = _hue(rgb)
    colorful = np.clip(s * 4, 0, 1) * np.clip(v * 4, 0, 1)
    hue_shift = _around_wheel(h, [row[0] for row in hsl]) * HUE_SHIFT_MAX
    sat = _around_wheel(h, [row[1] for row in hsl])
    lum = _around_wheel(h, [row[2] for row in hsl])
    h = h + hue_shift * colorful
    s = np.clip(s * (1 + sat * colorful), 0, 1)
    v = np.clip(v * (1 + 0.4 * lum * colorful * s), 0, 1)
    return _rgb(h, s, v)


def apply(rgb: np.ndarray, g: ColorGrade, user_lut: np.ndarray | None = None) -> np.ndarray:
    """색(0~1, 끝 차원 3)에 보정을 입힌 새 배열 (0~1). user_lut = 외부 LUT 표 (read_cube) — g.lut_strength만큼 섞는다."""
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    if g.exposure or g.temperature or g.tint:  # 선형 빛에서
        gains = np.array([1 + TEMPERATURE_GAIN * g.temperature, 1 - TINT_GAIN * g.tint, 1 - TEMPERATURE_GAIN * g.temperature],
                         np.float32)
        gains /= float(gains @ LUMA)  # 밝기는 그대로
        lin = x**GAMMA * gains * (2.0 ** (EXPOSURE_STOPS * g.exposure))
        x = np.clip(lin, 0.0, None) ** (1 / GAMMA)
    if g.highlights or g.shadows or g.whites or g.blacks or g.contrast or x.max() > 1:
        L = x @ LUMA
        x = x + (tone(L, g) - L)[..., None]  # 밝기만 옮긴다 (색은 그대로)
    if g.fade > 0:
        x = 0.12 * g.fade + x * (1 - 0.2 * g.fade)
    if g.saturation or g.vibrance:
        L = x @ LUMA
        chroma = x - L[..., None]
        now = x.max(axis=-1) - x.min(axis=-1)
        factor = (1 + g.saturation) * (1 + g.vibrance * (1 - np.clip(now * 1.6, 0, 1)))
        x = L[..., None] + chroma * np.maximum(factor, 0)[..., None]
    x = hsl_adjust(np.clip(x, 0.0, 1.0), g.hsl)
    if user_lut is not None and g.lut_strength > 0:
        x = x + (sample_lut(user_lut, x) - x) * min(1.0, g.lut_strength)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def lattice(size: int = LUT_SIZE) -> np.ndarray:
    """LUT 격자의 색들 (size, size, size, 3), [파랑][초록][빨강] 순 (.cube는 빨강이 가장 빨리 바뀐다)."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def build_lut(g: ColorGrade, user_lut: np.ndarray | None = None, size: int = LUT_SIZE) -> np.ndarray:
    """보정을 구운 LUT 표 (size, size, size, 3) [파랑][초록][빨강]."""
    return apply(lattice(size), g, user_lut)


def sample_lut(table: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """LUT 표로 색을 바꾼다 (삼선형 보간). rgb 0~1 (끝 차원 3) → 같은 모양."""
    n = table.shape[0]
    shape = np.shape(rgb)
    pos = np.clip(np.asarray(rgb, np.float32).reshape(-1, 3), 0, 1) * (n - 1)
    i0 = np.minimum(pos.astype(np.int32), n - 2)
    f = pos - i0
    base = (i0[:, 2] * n + i0[:, 1]) * n + i0[:, 0]  # 표를 한 줄로 편 자리 ([파랑][초록][빨강])
    flat = table.reshape(-1, 3)
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    out = np.zeros_like(pos)
    for db, wb in ((0, 1 - fb), (1, fb)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            w_bg = wb * wg
            for dr, wr in ((0, 1 - fr), (1, fr)):
                out += flat[base + ((db * n + dg) * n + dr)] * (w_bg * wr)
    return out.reshape(shape)


def apply_image(rgb8: np.ndarray, table: np.ndarray) -> np.ndarray:
    """uint8 그림 → LUT를 입힌 uint8 그림 (화면 미리보기용)."""
    return (sample_lut(table, rgb8.astype(np.float32) / 255.0) * 255.0 + 0.5).astype(np.uint8)


def auto_grade(rgb8: np.ndarray, base: ColorGrade = ColorGrade()) -> ColorGrade:
    """자동 맞춤: 화면 평균 색이 회색이 되게 색온도·색조를, 가운데 밝기가 알맞게 노출을 맞춘다 (나머지 값은 그대로)."""
    x = rgb8.reshape(-1, 3).astype(np.float32) / 255.0
    L = x @ LUMA
    mid = x[(L > 0.08) & (L < 0.92)]  # 너무 어둡거나 하얗게 날아간 곳은 빼고
    if len(mid) < 100:
        return base
    lin = (mid**GAMMA).mean(axis=0)
    r, g, b = (float(v) for v in lin)
    temperature = (b - r) / (TEMPERATURE_GAIN * (r + b)) if r + b > 1e-6 else 0.0
    neutral = (r * (1 + TEMPERATURE_GAIN * temperature) + b * (1 - TEMPERATURE_GAIN * temperature)) / 2
    tint = (1 - neutral / g) / TINT_GAIN if g > 1e-6 else 0.0
    median = float(np.median(L[(L > 0.02)])) if np.any(L > 0.02) else 0.45
    exposure = math.log2(max(1e-4, 0.45**GAMMA) / max(1e-4, median**GAMMA)) / EXPOSURE_STOPS
    clamp = lambda v, limit: float(max(-limit, min(limit, round(v, 2))))  # noqa: E731
    return replace(base, temperature=clamp(temperature, 0.8), tint=clamp(tint, 0.6), exposure=clamp(exposure, 0.5))


# ── .cube 파일 ──────────────────────────────────────────────────


def write_cube(path: Path, table: np.ndarray) -> Path:
    """LUT 표를 .cube로 (Adobe/Resolve 형식 — 다른 편집 프로그램에서도 쓸 수 있다)."""
    n = table.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(f'TITLE "KS Studio"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        np.savetxt(f, table.reshape(-1, 3), fmt="%.5f")
    tmp.replace(path)
    return path


def read_cube(path: Path) -> np.ndarray:
    """.cube 3D LUT → 표 (n, n, n, 3) [파랑][초록][빨강], 0~1. 읽을 수 없으면 ValueError."""
    size, domain_min, domain_max, rows = None, np.zeros(3), np.ones(3), []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "LUT_1D_SIZE":
            raise ValueError("1D LUT은 쓸 수 없습니다 (3D LUT만)")
        elif key == "DOMAIN_MIN":
            domain_min = np.array([float(v) for v in line.split()[1:4]])
        elif key == "DOMAIN_MAX":
            domain_max = np.array([float(v) for v in line.split()[1:4]])
        elif key[0].isdigit() or key[0] in "-.":
            rows.append([float(v) for v in line.split()[:3]])
        # TITLE 등 다른 줄은 넘긴다
    if size is None or size < 2 or len(rows) != size**3:
        raise ValueError(f"3D LUT 크기가 맞지 않습니다 ({len(rows)}줄, 크기 {size})")
    table = (np.array(rows, np.float32) - domain_min) / np.maximum(domain_max - domain_min, 1e-6)
    return np.clip(table, 0, 1).reshape(size, size, size, 3)


# ── 저장 (프로젝트의 color/settings.json) ─────────────────────────


AI_SUFFIX = "_보정.mp4"  # Workspace.ai_output: ai/<원본 이름>_보정.mp4


@dataclass(frozen=True)
class ColorState:
    grades: dict = field(default_factory=dict)  # 원본(Workspace.to_source) → ColorGrade
    imported: tuple[str, ...] = ()  # 색 보정 탭에서 가져온 영상

    def grade_for(self, source: str) -> ColorGrade | None:
        """그 영상의 색 보정 (없으면 None). AI 보정본(ai/<이름>_보정.mp4)은 원본의 색 보정을 따른다."""
        grade = self.grades.get(source)
        if grade is None:
            path = Path(source)
            if path.name.endswith(AI_SUFFIX) and path.parent.name == "ai":
                stem = path.name[: -len(AI_SUFFIX)]
                grade = next((g for s, g in self.grades.items() if Path(s).stem == stem), None)
        return None if grade is None or grade.is_identity else grade


def load_color_state(path: Path) -> ColorState:
    if not path.exists():
        return ColorState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        grades = {str(k): ColorGrade.from_dict(v) for k, v in data.get("grades", {}).items() if isinstance(v, dict)}
        return ColorState(grades, tuple(str(s) for s in data.get("imported", [])))
    except (OSError, ValueError, AttributeError, TypeError):
        return ColorState()


def save_color_state(path: Path, state: ColorState) -> None:
    write_json_atomic(path, {"version": 1, "grades": {k: g.to_dict() for k, g in state.grades.items() if not g.is_identity},
                             "imported": list(state.imported)})


def save_look(folder: Path, name: str, grade: ColorGrade) -> None:
    write_json_atomic(folder / f"{name.strip()}.json", {"version": 1, "grade": grade.look().to_dict()})


def load_look(folder: Path, name: str) -> ColorGrade:
    return ColorGrade.from_dict(json.loads((folder / f"{name}.json").read_text(encoding="utf-8")).get("grade", {}))
