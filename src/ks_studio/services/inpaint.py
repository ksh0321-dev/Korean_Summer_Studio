"""사람 뒤 배경 메꾸기 (몸매·얼굴 작게로 사람이 비켜난 자리). 메울 곳을 덮는 패치만 계산한다 — 화면 전체를 만들지 않는다.

- telea: 1/8 크기에서 둘레 배경으로 번져 채운다 (services.reshape.fill_telea, 모델 없음). 빠르지만 흐리다
- MI-GAN (Picsart, MIT, 28MB): 모바일용 GAN. 512 패치 하나에 약 20ms (RTX 5080)
- LaMa (big-lama, Apache 2.0, 208MB): 반복 무늬(나무결·벽지)를 잘 잇는다. 512 패치 하나에 약 38ms
- MemoryFill: 위의 무엇이든 감싸서 (1) 예전 프레임에서 본 진짜 배경을 기억해 쓰고 (2) 앞 프레임의 메운 결과와 섞어 떨림을 줄인다

실측 (4K 세로, 얼굴 작게 100%, 3초): 메운 곳의 프레임 사이 변화 TELEA 1.34 · MI-GAN 1.36 · LaMa 1.42 → 기억+섞기 1.22~1.25
(바로 옆 진짜 배경 0.34). 화질은 나무결·흰 벽 앞 머리카락에서 MI-GAN·LaMa가 눈에 띄게 낫다.
"""

import logging
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ks_studio.app.paths import MODELS_DIR

log = logging.getLogger(__name__)

INPAINT_DIR = MODELS_DIR / "inpaint"
FILL_MODELS = {  # 설정 값 → (이름, 파일, 주소, 크기 MB)
    "migan": ("MI-GAN", "migan_pipeline_v2.onnx", "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx", 28),
    "lama": ("LaMa", "lama_fp32.onnx", "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", 208),
}
FILLS = ("telea", "migan", "lama")
PATCH = 512  # 두 모델 모두 512×512에서 배웠다 (LaMa ONNX는 512 고정)
WORK_LONG = 1920  # 화면 긴 변을 이만큼으로 줄인 크기에서 패치를 뜬다 (4K → 1/2: 패치 수 1/4, 배경이라 충분)
CONTEXT = 48  # 패치 둘레 참고용 띠 (작업 크기 화소) — 가운데 칸만 붙인다
LAMA_BATCH = 2  # 한 번에 4장은 GPU 메모리가 모자랐다 (RTX 5080, 다른 모델과 같이)
CHUNK = 1 << 20


def fill_model_ready(kind: str, folder: Path = INPAINT_DIR) -> bool:
    path = folder / FILL_MODELS[kind][1]
    return path.is_file() and path.stat().st_size > 0


def download_fill_model(kind: str, on_progress=None, folder: Path = INPAINT_DIR, url: str | None = None) -> None:
    """받는 중엔 임시 이름이라 끊겨도 '받은 것'으로 보이지 않는다."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / FILL_MODELS[kind][1]
    part = target.with_name(target.name + ".part")
    try:
        request = urllib.request.Request(url or FILL_MODELS[kind][2], headers={"User-Agent": "KS-Studio"})
        with urllib.request.urlopen(request, timeout=30) as response, open(part, "wb") as out:
            total, done = int(response.headers.get("Content-Length") or 0), 0
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if on_progress and total:
                    on_progress(min(1.0, done / total))
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)


_sessions: dict[str, object] = {}
_sessions_lock = threading.Lock()


def _session(kind: str, folder: Path):
    """모델 세션은 하나만 (샘플 컷·미리보기·보정본 렌더 스레드가 같이 쓴다 — run은 여러 스레드에서 불러도 된다)."""
    with _sessions_lock:
        if kind not in _sessions:
            from ks_studio.services.accel import onnx_session

            # MI-GAN은 입력 크기가 바뀌는 모델이라 Core ML(맥)에 맞지 않는다 → 맥에선 CPU, LaMa(512 고정)는 Core ML
            _sessions[kind] = onnx_session(folder / FILL_MODELS[kind][1], coreml=kind != "migan")
        return _sessions[kind]


def _run_model(kind: str, folder: Path, windows: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    """512 패치들 → 메운 패치들 (RGB uint8). masks: 1 = 메울 곳."""
    session = _session(kind, folder)
    if kind == "migan":  # 한 장씩 (묶음 입력은 안 됐다), 마스크 255 = 아는 곳
        return [session.run(None, {"image": w.transpose(2, 0, 1)[None], "mask": ((1 - m) * 255)[None, None].astype(np.uint8)})[0][0]
                .transpose(1, 2, 0) for w, m in zip(windows, masks)]
    results = []
    for i in range(0, len(windows), LAMA_BATCH):  # 0~1 그림, 1 = 메울 곳, 결과는 0~255
        group_w, group_m = windows[i : i + LAMA_BATCH], masks[i : i + LAMA_BATCH]
        count = len(group_w)
        # 묶음 크기를 늘 같게 (모자라면 앞 것을 되풀이): 크기가 바뀌면 GPU가 계산 방법을 다시 찾아 4배 넘게 느렸다 (실측)
        group_w, group_m = group_w + group_w[:1] * (LAMA_BATCH - count), group_m + group_m[:1] * (LAMA_BATCH - count)
        image = (np.stack(group_w).transpose(0, 3, 1, 2) / 255).astype(np.float32)
        mask = np.stack(group_m)[:, None].astype(np.float32)
        out = session.run(None, {"image": image, "mask": mask})[0][:count]
        results += list(np.clip(out, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1))
    return results


class PatchFill:
    """메울 곳(need)을 덮는 512 패치들만 모델로 메꾼다. reshape.fill_telea와 같은 모양으로 부른다.

    패치는 화면을 긴 변 WORK_LONG으로 줄인 크기에서 뜬다. 각 패치는 가운데 칸만 붙이고 둘레(CONTEXT)는 참고용 —
    가운데 칸들이 겹치지 않아 한 화소는 한 패치에서만 온다. 사람 자리(hole) 전체를 '모르는 곳'으로 줘서
    모델이 사람 화소를 배경으로 베끼지 않게 한다.
    """

    def __init__(self, kind: str, folder: Path = INPAINT_DIR, run=None) -> None:
        self.kind, self._folder = kind, folder
        self._run = run or (lambda windows, masks: _run_model(kind, folder, windows, masks))
        self.patches = 0  # 지금까지 메운 패치 수 (확인용)

    def __call__(self, image, box, hole, need, carry) -> np.ndarray:
        x0, y0, x1, y1 = box
        h, w = image.shape[:2]
        out = image[y0:y1, x0:x1].copy()
        rows, cols = np.nonzero(need)
        if not len(rows):
            return out
        scale = min(1.0, WORK_LONG / max(h, w))
        win = min(round(PATCH / scale), h, w)  # 원래 크기에서 패치 한 변
        inner = max(16, win - 2 * round(CONTEXT / scale))  # 붙이는 가운데 칸
        tasks, windows, masks = [], [], []
        for ty in range(y0 + rows.min(), y0 + rows.max() + 1, inner):
            for tx in range(x0 + cols.min(), x0 + cols.max() + 1, inner):
                if not need[max(0, ty - y0) : ty - y0 + inner, max(0, tx - x0) : tx - x0 + inner].any():
                    continue
                wy = min(max(0, ty - (win - inner) // 2), h - win)  # 패치 창 (화면 안으로 밀어 넣는다)
                wx = min(max(0, tx - (win - inner) // 2), w - win)
                mask = np.zeros((win, win), np.uint8)  # 패치 창 안의 사람 자리 (box 밖은 움직이지 않은 곳 = 아는 곳)
                iy0, ix0 = max(wy, y0), max(wx, x0)
                iy1, ix1 = min(wy + win, y1), min(wx + win, x1)
                mask[iy0 - wy : iy1 - wy, ix0 - wx : ix1 - wx] = hole[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
                windows.append(cv2.resize(image[wy : wy + win, wx : wx + win], (PATCH, PATCH), interpolation=cv2.INTER_AREA))
                masks.append((cv2.resize(mask, (PATCH, PATCH), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))
                tasks.append((wy, wx, ty, tx))
        self.patches += len(tasks)
        for (wy, wx, ty, tx), result in zip(tasks, self._run(windows, masks)):
            ay0, ax0 = max(ty, y0, wy), max(tx, x0, wx)  # 가운데 칸 ∩ box
            ay1, ax1 = min(ty + inner, y1, wy + win), min(tx + inner, x1, wx + win)
            if ay1 <= ay0 or ax1 <= ax0:
                continue
            big = cv2.resize(result, (win, win), interpolation=cv2.INTER_LINEAR)
            out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = big[ay0 - wy : ay1 - wy, ax0 - wx : ax1 - wx]
        return out


def make_fill(kind: str):
    """설정의 메꾸기 → 메꾸기 함수. 모델을 아직 안 받았으면 None (reshape가 telea로 한다)."""
    if kind in FILL_MODELS and fill_model_ready(kind):
        return PatchFill(kind)
    return None


class MemoryFill:
    """다른 메꾸기를 감싸 영상을 덜 일렁이게: (1) 예전에 본 진짜 배경 기억 (2) 앞 프레임의 메운 결과와 섞기.

    계산은 메울 화소(need)에서만 한다 (4K에서 보통 10~20만 화소). 기억 고치기는 메울 영역 둘레만, UPDATE_EVERY장마다 —
    배경은 거의 그대로라 충분하다. 실험 때 4K 화면 전체로 했더니 프레임당 0.5초, 영역 전체로 해도 0.07초가 더 걸렸다.
    카메라가 움직이면 (보이는 배경이 기억과 많이 다르면) 기억을 버린다 — 손으로 든 영상은 기억을 거의 못 쓴다.
    """

    UPDATE_EVERY = 5
    MODEL_EVERY = 3  # 모델은 3장마다 — 사이 프레임은 앞 프레임에서 메운 값을 그대로 (배경은 그대로라 같은 값이 나와야 맞다)
    COVERED = 0.98  # 사이 프레임이라도 메울 곳의 2% 넘게가 처음 보는 자리면 모델을 다시 부른다 (머리가 움직였다)

    def __init__(self, base, blend: float = 0.35, moved_limit: float = 10.0, margin: float = 0.15) -> None:
        self.base, self.blend, self.moved_limit, self.margin = base, blend, moved_limit, margin
        self.plate = self.known = self.previous = self.previous_need = None
        self._previous_at = None  # 앞 프레임에서 메운 화소 자리 (ys, xs)
        self.resets = self.calls = self.model_calls = 0  # 카메라가 움직여 기억을 버린 횟수 / 부른 횟수 / 모델을 부른 횟수 (확인용)
        self.remembered = 0  # 기억으로 채운 화소 수 (확인용)

    def _remember(self, image, box, carry) -> None:
        """메울 영역 둘레에서 지금 보이는 배경을 기억에 넣는다 (카메라가 움직였으면 기억을 먼저 버린다)."""
        from ks_studio.services.reshape import GRID

        h, w = image.shape[:2]
        x0, y0, x1, y1 = box
        pad = round(self.margin * max(x1 - x0, y1 - y0))  # 다음 프레임에 머리가 조금 움직여도 쓰게
        ax0, ay0 = max(0, x0 - pad) // GRID * GRID, max(0, y0 - pad) // GRID * GRID
        ax1, ay1 = min(w, x1 + pad), min(h, y1 + pad)
        cells = carry[ay0 // GRID : (ay1 - 1) // GRID + 2, ax0 // GRID : (ax1 - 1) // GRID + 2]
        background = cv2.resize(cells, (cells.shape[1] * GRID, cells.shape[0] * GRID))[: ay1 - ay0, : ax1 - ax0] < 0.01
        area = np.s_[ay0:ay1, ax0:ax1]
        plate, known, now = self.plate[area], self.known[area], image[area]
        both = (background & known)[::4, ::4]  # 지금도 보이고 기억에도 있는 배경끼리 (4칸마다 하나)
        if both.any() and float(np.abs(plate[::4, ::4][both].astype(np.int16) - now[::4, ::4][both]).mean()) > self.moved_limit:
            self.known[:] = False
            self.previous_need[:] = False
            self.resets += 1
        np.copyto(plate, now, where=background[..., None])
        known |= background

    def __call__(self, image, box, hole, need, carry) -> np.ndarray:
        from ks_studio.services.reshape import fill_telea

        h, w = image.shape[:2]
        if self.plate is None or self.plate.shape != image.shape:
            self.plate, self.known = np.zeros_like(image), np.zeros((h, w), bool)
            self.previous, self.previous_need = np.zeros_like(image), np.zeros((h, w), bool)
        if self.calls % self.UPDATE_EVERY == 0:
            self._remember(image, box, carry)
        x0, y0 = box[:2]
        ys, xs = np.nonzero(need)
        gy, gx = ys + y0, xs + x0  # 화면 좌표
        remembered = self.known[gy, gx]
        again = self.previous_need[gy, gx] & ~remembered  # 앞 프레임에서도 메운 곳
        covered = float((again | remembered).mean()) if len(ys) else 1.0
        fresh = self.calls % self.MODEL_EVERY == 0 or covered < self.COVERED
        self.calls += 1
        self.model_calls += fresh
        # 사이 프레임: 처음 보는 몇 화소만 빠른 telea로 (앞 값을 쓰는 곳이 거의 다다)
        out = (self.base or fill_telea)(image, box, hole, need, carry) if fresh else fill_telea(image, box, hole, need, carry)
        values = out[ys, xs]
        if again.any():
            previous = self.previous[gy[again], gx[again]]
            if fresh:  # 새로 메운 값과 앞 값을 섞는다
                values[again] = (self.blend * values[again] + (1 - self.blend) * previous + 0.5).astype(np.uint8)
            else:
                values[again] = previous
        values[remembered] = self.plate[gy[remembered], gx[remembered]]
        out[ys, xs] = values
        self.remembered += int(remembered.sum())
        if self._previous_at is not None:
            self.previous_need[self._previous_at] = False
        self.previous[gy, gx] = values
        self.previous_need[gy, gx] = True
        self._previous_at = (gy, gx)
        return out
