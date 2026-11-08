"""yuv420p 프레임의 Y·U·V 판을 numpy로 꺼내고 다시 넣기.

frame.to_ndarray()는 세 판을 (높이×1.5, 폭) 하나로 이어 붙이는데, 높이의 절반이 홀수(예: 1422 → 711)면
U·V 판이 줄 경계에 맞지 않아 잘라 쓸 수 없다 (실측: 9:16 저해상도 미리보기 800×1422에서 깨짐) → 판을 따로 다룬다.
"""

import threading
from fractions import Fraction

import av
import av.filter
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


# ── 정확한 색 형식 바꾸기 ──────────────────────────────────────────
# PyAV의 reformat·to_ndarray(format=...)는 FFmpeg 변환기(swscale) 기본값이라 반올림 대신 버림을 한다:
# yuv → rgb → yuv를 한 번 거치면 화면 전체가 255단계 중 약 1단계 어두워졌다 (실측 -1.10). accurate_rnd를 켜면 -0.001.
# 색 보정·화면 속 화면·디졸브·비율이 다른 클립의 여백처럼 RGB로 바꿔 그리는 곳은 모두 이것으로 바꾼다.
SWS_FLAGS = "bicubic+accurate_rnd+full_chroma_int"
# 색 규격도 오갈 때 같게: 원본에 BT.709 표시가 있으면 FFmpeg은 yuv → rgb에 709를, 되돌릴 땐(rgb엔 표시가 없다) 601을 써서
# 화면이 255단계 중 1.4단계 밝아졌다 (실측, 아이폰 영상). 규격은 영상 표시를 따르고, 표시가 없으면 플레이어들처럼
# 세로 720 이상(HD)은 709, 그보다 작으면(SD) 601로 본다. rgb → yuv는 만드는 크기로 같은 규칙.
RGB_FORMATS = ("rgb24", "bgr24", "gbrp", "rgba", "bgra")
# yuv → yuv(크기만 바꾸기): 들어올 때와 나갈 때 규격을 같게 못 박는다. 안 그러면 709 표시가 붙은 프레임을 표시 없는 출력으로
# 바꾸면서 601로 다시 계산해 화면이 1~1.5단계 어두워졌다 (실측 — 효과 없이 내보내도 아이폰 영상에서 생겼다)
KEEP_MATRIX = "in_color_matrix=bt709:out_color_matrix=bt709"
_MATRICES = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020", 10: "bt2020"}  # AVColorSpace → scale 필터 이름


def matrix(colorspace, height: int) -> str:
    """영상 표시(colorspace)에 맞는 색 규격. 표시가 없으면 HD는 709, SD는 601."""
    try:
        known = _MATRICES.get(int(colorspace))
    except (TypeError, ValueError):
        known = None
    return known or ("bt709" if height >= 720 else "bt601")


_local = threading.local()  # 필터 그래프는 스레드마다 (재생·내보내기가 동시에 돈다)


def scale_args(source: str, target: str, width: int | None = None, height: int | None = None,
               colorspace=None, source_height: int = 1080) -> str:
    """scale 필터 인자: 크기 · 정확한 반올림 · 색 규격 (rgb ↔ yuv는 영상 표시대로, yuv → yuv는 그대로)."""
    args = [f"{width}:{height}"] if width and height else []
    args.append(f"flags={SWS_FLAGS}")
    if source not in RGB_FORMATS and target in RGB_FORMATS:
        args.append(f"in_color_matrix={matrix(colorspace, source_height)}")
    elif source in RGB_FORMATS and target not in RGB_FORMATS:
        args.append(f"out_color_matrix={matrix(None, height or source_height)}:out_range=tv")
    elif source not in RGB_FORMATS and target not in RGB_FORMATS:
        args.append(KEEP_MATRIX)
    return ":".join(args)


def convert(frame: av.VideoFrame, format: str, width: int | None = None, height: int | None = None) -> av.VideoFrame:
    """frame을 format(·크기)으로. 같은 형식·크기면 그대로. pts·time_base는 그대로 둔다."""
    width, height = width or frame.width, height or frame.height
    source = frame.format.name
    if source == format and (width, height) == (frame.width, frame.height):
        return frame
    graphs = getattr(_local, "graphs", None)
    if graphs is None:
        graphs = _local.graphs = {}
    space = frame.colorspace if source not in RGB_FORMATS else None
    key = (frame.width, frame.height, source, width, height, format, int(space or 0))
    graph = graphs.get(key)
    if graph is None:
        if len(graphs) > 32:  # 크기가 자꾸 바뀌는 경우(창 크기 조절) 쌓이지 않게
            graphs.clear()
        graph = av.filter.Graph()
        buffer = graph.add_buffer(width=frame.width, height=frame.height, format=source, time_base=Fraction(1, 30))
        graph.link_nodes(buffer, graph.add("scale", scale_args(source, format, width, height, space, frame.height)), graph.add("format", format),
                         graph.add("buffersink")).configure()
        graphs[key] = graph
    pts, time_base = frame.pts, frame.time_base
    frame.pts = 0
    graph.push(frame)
    out = graph.pull()
    frame.pts = out.pts = pts
    if time_base is not None:
        out.time_base = time_base
    return out


def to_rgb(frame: av.VideoFrame, width: int | None = None, height: int | None = None) -> np.ndarray:
    return convert(frame, "rgb24", width, height).to_ndarray()


def from_rgb(rgb: np.ndarray, format: str, width: int | None = None, height: int | None = None) -> av.VideoFrame:
    return convert(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24"), format, width, height)
