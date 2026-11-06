"""yuv420p 프레임의 Y·U·V 판을 numpy로 꺼내고 다시 넣기.

frame.to_ndarray()는 세 판을 (높이×1.5, 폭) 하나로 이어 붙이는데, 높이의 절반이 홀수(예: 1422 → 711)면
U·V 판이 줄 경계에 맞지 않아 잘라 쓸 수 없다 (실측: 9:16 저해상도 미리보기 800×1422에서 깨짐) → 판을 따로 다룬다.
"""

import av
import numpy as np


def planes(frame: av.VideoFrame) -> list[np.ndarray]:
    """[Y(h×w), U(h/2×w/2), V(h/2×w/2)] 복사본. 판마다 줄 끝 여백(line_size)은 잘라 낸다."""
    out = []
    for plane in frame.planes:
        rows = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
        out.append(rows[:, : plane.width].copy())
    return out


def from_planes(y: np.ndarray, u: np.ndarray, v: np.ndarray, like: av.VideoFrame) -> av.VideoFrame:
    """세 판으로 새 yuv420p 프레임 (pts·time_base는 like에서)."""
    frame = av.VideoFrame(y.shape[1], y.shape[0], "yuv420p")
    for plane, data in zip(frame.planes, (y, u, v)):
        buffer = np.zeros((plane.height, plane.line_size), np.uint8)
        buffer[:, : data.shape[1]] = data
        plane.update(buffer)
    frame.pts = like.pts
    if like.time_base is not None:
        frame.time_base = like.time_base
    return frame
