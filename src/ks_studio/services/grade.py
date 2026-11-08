"""색 보정을 영상에 입힌다: 보정값 → .cube 파일(캐시) → FFmpeg lut3d (C로 돌아 4K도 빠르다).

같은 보정값이면 같은 파일 (이름 = 값의 해시). 타임라인 미리보기·내보내기·AI 보정본이 이 파일을 같이 쓴다.
"""

import hashlib
import json
import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ks_studio.core.grade import ColorGrade, ColorState, build_lut, load_color_state, read_cube, write_cube
from ks_studio.core.workspace import Workspace

log = logging.getLogger(__name__)

CUBE_DIR = "grades"  # cache/grades/<해시>.cube
KEEP_CUBES = 40  # 이보다 많이 쌓이면 오래 안 쓴 것부터 지운다 (값을 바꿀 때마다 새 파일이 생긴다)
_states: dict[Path, tuple[float, ColorState]] = {}  # settings.json → (수정 시각, 읽은 것) — 타임라인 렌더러를 만들 때마다 다시 읽지 않게


def color_state(workspace: Workspace) -> ColorState:
    path = workspace.color_settings_file
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ColorState()
    cached = _states.get(path)
    if cached is None or cached[0] != mtime:
        cached = (mtime, load_color_state(path))
        _states[path] = cached
    return cached[1]


def user_lut(workspace: Workspace, grade: ColorGrade) -> np.ndarray | None:
    """보정에 붙인 외부 .cube 표 (없거나 못 읽으면 None — 경고만)."""
    if not grade.lut or grade.lut_strength <= 0:
        return None
    try:
        return read_cube(workspace.resolve(grade.lut))
    except (OSError, ValueError) as e:
        log.warning("LUT을 읽을 수 없음 %s: %s", grade.lut, e)
        return None


def cube_for(workspace: Workspace, grade: ColorGrade | None) -> Path | None:
    """그 보정을 구운 .cube 파일 (없으면 만든다). 보정이 없으면 None."""
    if grade is None or grade.is_identity:
        return None
    key = json.dumps(grade.to_dict(), sort_keys=True)
    if grade.lut:  # 외부 LUT 파일을 바꿔 끼우면 새로 굽는다
        try:
            key += str(workspace.resolve(grade.lut).stat().st_mtime_ns)
        except OSError:
            pass
    folder = workspace.cache_dir / CUBE_DIR
    path = folder / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.cube"
    if not path.exists():
        write_cube(path, build_lut(grade, user_lut(workspace, grade)))
        old = sorted(folder.glob("*.cube"), key=lambda p: p.stat().st_mtime, reverse=True)[KEEP_CUBES:]
        for stale in old:
            stale.unlink(missing_ok=True)
    return path


def source_cube(workspace: Workspace, source: str) -> Path | None:
    """원본(또는 그 AI 보정본)에 입힐 .cube 파일. 색 보정이 없으면 None."""
    try:
        return cube_for(workspace, color_state(workspace).grade_for(source))
    except OSError as e:
        log.warning("색 보정 LUT을 만들 수 없음 %s: %s", source, e)
        return None


class LutFilter:
    """.cube 하나를 프레임에 입힌다 (FFmpeg lut3d). 크기·형식마다 필터 그래프 하나 (다시 만들지 않게)."""

    def __init__(self, cube: Path) -> None:
        self.cube = cube
        self._graphs: dict[tuple, av.filter.Graph] = {}

    def frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        """같은 형식(yuv420p 등)으로 돌려준다. 못 쓰면 그대로 (경고만)."""
        key = (frame.width, frame.height, frame.format.name)
        try:
            graph = self._graphs.get(key)
            if graph is None:
                graph = av.filter.Graph()
                source = graph.add_buffer(width=frame.width, height=frame.height, format=frame.format.name,
                                          time_base=Fraction(1, 30))
                # 경로는 이름 붙인 옵션으로 (필터 글로 쓰면 C: 같은 콜론을 따로 막아야 한다)
                graph.link_nodes(source, graph.add("lut3d", file=str(self.cube)), graph.add("format", frame.format.name),
                                 graph.add("buffersink")).configure()
                self._graphs[key] = graph
            pts, time_base = frame.pts, frame.time_base
            frame.pts = 0
            graph.push(frame)
            out = graph.pull()
            out.pts = pts
            if time_base is not None:
                out.time_base = time_base
            return out
        except (av.FFmpegError, ValueError) as e:
            log.warning("색 보정을 입힐 수 없음 %s: %s", self.cube, e)
            self._graphs.pop(key, None)
            return frame

    def rgb(self, rgb: np.ndarray) -> np.ndarray:
        return self.frame(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")).to_ndarray(format="rgb24")
