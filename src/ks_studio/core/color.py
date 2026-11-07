"""클립 그림 다듬기: 자르기·회전·뒤집기와 색 보정(밝기·대비·채도·색온도)·필터. RGB uint8 배열에 — 순수 numpy.

- 밝기·대비는 256칸 표(LUT) 하나로 (4K에서도 빠르다), 채도는 밝기와의 차이를 늘이고 줄이고, 색온도는 빨강·파랑 비율을 바꾼다
- 필터(LOOKS)는 슬라이더 값 묶음 + 조금의 곡선. 클립 자신의 슬라이더 값에 더해진다
- .cube LUT 파일은 FFmpeg lut3d 필터로 (services.render) — 여기는 슬라이더·필터만
"""

import numpy as np

LOOKS = {  # 키 → (이름, 밝기, 대비, 채도, 색온도, 검정 띄우기)
    "": ("없음", 0.0, 0.0, 0.0, 0.0, 0.0),
    "warm": ("따뜻하게", 0.02, 0.05, 0.08, 0.45, 0.0),
    "cool": ("시원하게", 0.02, 0.05, 0.0, -0.45, 0.0),
    "vivid": ("선명하게", 0.0, 0.18, 0.35, 0.05, 0.0),
    "soft": ("부드럽게", 0.06, -0.2, -0.08, 0.1, 0.04),
    "film": ("필름", 0.0, -0.12, -0.22, 0.18, 0.08),
    "mono": ("흑백", 0.0, 0.12, -1.0, 0.0, 0.0),
}
BRIGHT_MAX = 64  # 밝기 ±1 = 화소값 ±64
CONTRAST_MAX = 0.6  # 대비 ±1 = 기울기 ×(1 ± 0.6)
TEMP_MAX = 0.14  # 색온도 ±1 = 빨강 ×(1 ± 0.14), 파랑 ×(1 ∓ 0.14)


def shape(rgb: np.ndarray, crop: tuple[float, float, float, float], rotation: int, flip: bool) -> np.ndarray:
    """자르기(원본 비율) → 회전(시계 방향) → 좌우 뒤집기."""
    h, w = rgb.shape[:2]
    left, top, right, bottom = (min(0.45, max(0.0, v)) for v in crop)
    x0, y0 = round(w * left), round(h * top)
    x1, y1 = max(x0 + 2, w - round(w * right)), max(y0 + 2, h - round(h * bottom))
    out = rgb[y0:y1, x0:x1]
    turns = (rotation // 90) % 4
    if turns:
        out = np.rot90(out, -turns)  # rot90은 반시계 → 음수
    if flip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def grade(rgb: np.ndarray, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0,
          temperature: float = 0.0, look: str = "") -> np.ndarray:
    """색 보정한 새 배열 (아무것도 안 바꾸면 그대로 돌려준다)."""
    _, lb, lc, ls, lt, lift = LOOKS.get(look, LOOKS[""])
    b, c, s, t = brightness + lb, contrast + lc, max(-1.0, saturation + ls), temperature + lt
    if not (b or c or s or t or lift):
        return rgb
    x = np.arange(256, dtype=np.float32)
    curve = (x - 128) * (1 + c * CONTRAST_MAX) + 128 + b * BRIGHT_MAX
    curve = lift * 255 + curve * (1 - lift)  # 검정을 띄워 필름처럼
    lut = np.clip(curve + 0.5, 0, 255).astype(np.uint8)
    out = np.take(lut, rgb)  # OpenCV 없이 (기본 설치엔 AI 패키지가 없다)
    if s or t:
        work = out.astype(np.float32)
        if s:
            gray = work @ np.array([0.299, 0.587, 0.114], np.float32)
            work = gray[..., None] + (work - gray[..., None]) * (1 + s)
        if t:
            work *= np.array([1 + t * TEMP_MAX, 1 + t * TEMP_MAX * 0.2, 1 - t * TEMP_MAX], np.float32)
        out = np.clip(work + 0.5, 0, 255).astype(np.uint8)
    return out
