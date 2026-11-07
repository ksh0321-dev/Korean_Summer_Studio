"""다른 사람 얼굴 흐리기: 프레임마다 찾은 얼굴들 → 흐릴 상자. 순수 numpy (모델·영상 없음).

- 얼굴 인식값만으로 프레임마다 따로 판단하면 옆얼굴·흔들린 얼굴에서 틀린다 (주인공이 흐려지거나 남이 안 흐려짐)
  → 앞뒤 프레임의 얼굴을 상자 겹침(IoU)으로 이어 '같은 얼굴 줄기'로 묶고, 줄기 전체의 닮은 정도로 판단한다
- 주인공은 얼굴 점(랜드마크)으로 따로 쫓고 있다 → 그 자리와 겹치는 줄기는 흐리지 않는다
- 몇 프레임 얼굴을 놓쳐도(움직임·가림) 흐림이 깜빡 풀리지 않게 앞뒤 HOLD장을 이어 둔다
"""

from dataclasses import dataclass

import numpy as np

from ks_studio.core.people import SAME_PERSON, similarity

LINK_IOU = 0.3  # 다음 프레임 얼굴과 이 이상 겹치면 같은 얼굴
HOLD = 6  # 얼굴을 놓친 프레임을 이만큼까지 앞뒤로 메운다 (60fps에서 0.1초)
GROW = 0.3  # 흐릴 상자는 얼굴 상자보다 사방으로 30% 크게 (머리카락·귀까지)
BLUR_STYLES = {"mosaic": "모자이크", "blur": "흐림"}


@dataclass(frozen=True)
class SeenFace:
    frame: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 (화면 좌표)
    embedding: np.ndarray  # (512,)


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link(faces: list[SeenFace]) -> list[list[SeenFace]]:
    """얼굴들을 줄기로 잇는다: 가장 가까운 앞 프레임(HOLD장 안)의 줄기 끝과 가장 많이 겹치는 것에. 프레임 순서대로."""
    tracks: list[list[SeenFace]] = []
    for frame in sorted({f.frame for f in faces}):
        here = [f for f in faces if f.frame == frame]
        open_tracks = [t for t in tracks if 0 < frame - t[-1].frame <= HOLD + 1]
        pairs = sorted(((iou(t[-1].box, f.box), ti, fi) for ti, t in enumerate(open_tracks) for fi, f in enumerate(here)), reverse=True)
        used_t, used_f = set(), set()
        for score, ti, fi in pairs:
            if score < LINK_IOU or ti in used_t or fi in used_f:
                continue
            open_tracks[ti].append(here[fi])
            used_t.add(ti)
            used_f.add(fi)
        tracks += [[f] for fi, f in enumerate(here) if fi not in used_f]
    return tracks


def _grown(box, grow: float = GROW) -> tuple[float, float, float, float]:
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - w * grow, box[1] - h * grow, box[2] + w * grow, box[3] + h * grow


def blur_plan(faces: list[SeenFace], keep: list[np.ndarray], protected: dict[int, tuple] | None = None,
              hold: int = HOLD) -> dict[int, list[tuple[float, float, float, float]]]:
    """프레임 → 흐릴 상자들.

    keep = 흐리지 않을 사람들의 인식값 (주인공 + 이 영상에서 고른 사람 + 등록한 사람).
    protected = 프레임 → 주인공 얼굴 상자 (얼굴 점으로 쫓은 것). 그와 겹치는 줄기는 흐리지 않는다.
    """
    plan: dict[int, list] = {}
    for track in link(faces):
        if keep:
            score = float(np.mean([max(similarity(f.embedding, k) for k in keep) for f in track]))
            if score >= SAME_PERSON:
                continue
        if protected:
            overlaps = sum(1 for f in track if f.frame in protected and iou(f.box, protected[f.frame]) >= LINK_IOU)
            if overlaps * 2 >= len(track):
                continue
        by_frame = {f.frame: f.box for f in track}
        first, last = track[0].frame, track[-1].frame
        for frame in range(first - hold, last + hold + 1):  # 앞뒤로 조금 더, 사이에 놓친 프레임은 가까운 상자로
            if frame < 0:
                continue
            box = by_frame.get(frame)
            if box is None:
                nearest = min(by_frame, key=lambda f: abs(f - frame))
                if abs(nearest - frame) > hold and first <= frame <= last:
                    continue  # 줄기 가운데 오래 빈 곳 (화면 밖으로 나갔다 들어옴)
                box = by_frame[nearest]
            plan.setdefault(frame, []).append(_grown(box))
    return plan


def landmark_boxes(track: np.ndarray, start: int = 0) -> dict[int, tuple]:
    """주인공 얼굴 점 (프레임, 106, 2) → 프레임 → 얼굴 상자. 그 사람이 없는 프레임(NaN)은 빠진다."""
    boxes = {}
    for i, lm in enumerate(track):
        if not np.isnan(lm).any():
            boxes[start + i] = (float(lm[:, 0].min()), float(lm[:, 1].min()), float(lm[:, 0].max()), float(lm[:, 1].max()))
    return boxes
