"""내보낼 때 AI 보정 입히기: 타임라인에 쓰인 원본 프레임에만, 내보낼 크기로.

예전에는 AI 보정 탭에서 원본 전체를 원본 크기로 보정한 파일(ai/<이름>_보정.mp4)을 만들고, 그걸 다시 잘라 넣어야 했다.
실측 (4K 60fps 81초 → 54.8초 쇼츠): 얼굴 분석 4분 + 보정 렌더링 31분. 쓰지도 않는 27초까지, 4K로.
이제는 AI 보정 탭의 설정(원본마다)을 그대로 두고, 컷편집 타임라인을 내보낼 때 그 프레임에만 입힌다:

- 프레임마다: 원본 프레임 → **AI 보정** → 색 보정 → 자르기·회전·클립 색 → 화면 합치기(위 줄·디졸브) → 글자·자막
  (보정은 원본 프레임에 입힌다: 합친 화면에 입히면 위 줄 클립·디졸브의 겹친 얼굴까지 바뀌고, 편집할 때마다 다시 분석해야 한다.
   글자·자막은 맨 마지막에 굽기 때문에 몸매 보정이 글자를 휘게 하지 않는다)
- 보정은 내보낼 크기로 줄인 프레임에 (4K → 1080p면 화소 1/4). 얼굴 점은 원본 좌표라 같은 비율로 줄인다
- 얼굴 분석은 타임라인이 쓰는 원본 구간만 (앞뒤 떨림 줄이기 몫으로 몇 프레임 더). 분석한 프레임은 캐시에 쌓여서 다시 내보낼 때는 분석 없이
  — AI 보정 탭에서 영상 전체를 분석해 두었으면 그것을 쓴다
- 설정을 쓰는 조건: 샘플 컷을 보고 OK했고(approved) 효과가 있거나 다른 얼굴 흐리기를 켰다

컷편집 미리보기(재생)에는 입히지 않는다 (느리다 — 보정한 모습은 AI 보정 탭에서 본다). 내보내기의 저해상도 미리보기에는 입힌다.
"""

import logging
import os
from pathlib import Path

import av
import numpy as np

from ks_studio.core.beauty import BeautySettings, load_ai_state, smooth_track
from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.sources import SourceCache, cache_name
from ks_studio.services.yuv import convert

log = logging.getLogger(__name__)

SMOOTH_MARGIN = 3  # 분석할 구간 앞뒤 여유 (smooth_track의 반경 2 + 1)
SPANS_SUFFIX = "spans.npz"  # cache/<키>.faces_<사람>.spans.npz — 일부 구간만 분석한 얼굴 점 (프로젝트 정리가 남긴다)


def spans_file(track_path: Path) -> Path:
    """AI 보정 탭의 영상 전체 분석 파일(track_path) 옆에, 일부 구간만 분석한 것."""
    return track_path.with_name(track_path.name.removesuffix(".npy") + f".{SPANS_SUFFIX}")


def active_settings(workspace: Workspace) -> dict[str, BeautySettings]:
    """내보낼 때 입힐 원본별 AI 보정 설정 (Clip.source → 설정)."""
    state = load_ai_state(workspace.ai_settings_file)
    return {source: s for source, s in state.settings.items() if s.approved and s.needs_render}


def needed_ranges(project: Project, sources: set[str]) -> dict[str, list[tuple[int, int]]]:
    """원본별로 타임라인이 쓰는 원본 프레임 구간 [시작, 끝) — 겹치면 합치고 앞뒤 SMOOTH_MARGIN 여유."""
    found: dict[str, list[tuple[int, int]]] = {}
    for placed in project.placed():
        clip = placed.clip
        if clip.source in sources:
            found.setdefault(clip.source, []).append((max(0, clip.in_frame - SMOOTH_MARGIN), clip.out_frame + SMOOTH_MARGIN))
    merged: dict[str, list[tuple[int, int]]] = {}
    for source, ranges in found.items():
        out: list[list[int]] = []
        for a, b in sorted(ranges):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        merged[source] = [(a, b) for a, b in out]
    return merged


def frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return round(float(container.duration or 0) / 1_000_000 * float(stream.average_rate or 30))


class SpanTrack:
    """한 원본의 얼굴 점: 영상 길이만큼 (분석 안 한 프레임은 NaN) + 분석한 프레임 표시 + 그 프레임들의 모든 얼굴(흐리기용)."""

    def __init__(self, count: int) -> None:
        self.landmarks = np.full((count, 106, 2), np.nan, np.float32)
        self.done = np.zeros(count, bool)
        self.everyone: list = []

    @classmethod
    def load(cls, path: Path, count: int) -> "SpanTrack":
        track = cls(count)
        try:
            with np.load(path) as data:
                n = min(count, len(data["done"]))
                track.landmarks[:n] = data["landmarks"][:n]
                track.done[:n] = data["done"][:n]
                if "frames" in data.files:
                    from ks_studio.core.face_blur import SeenFace

                    track.everyone = [SeenFace(int(f), tuple(float(v) for v in b), e.astype(np.float32))
                                      for f, b, e in zip(data["frames"], data["boxes"], data["embeddings"])]
        except (OSError, ValueError, KeyError) as e:
            log.warning("얼굴 분석 캐시를 못 읽음 (다시 분석) %s: %s", path, e)
            return cls(count)
        return track

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name.removesuffix(".npz") + ".part.npz")
        np.savez_compressed(
            tmp, landmarks=self.landmarks, done=self.done,
            frames=np.array([f.frame for f in self.everyone], np.int32),
            boxes=np.array([f.box for f in self.everyone], np.float32).reshape(-1, 4),
            embeddings=np.array([f.embedding for f in self.everyone], np.float16).reshape(-1, 512),
        )
        os.replace(tmp, path)

    def missing(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """ranges 중 아직 분석하지 않은 구간들."""
        out = []
        for a, b in ranges:
            b = min(b, len(self.done))
            i = a
            while i < b:
                if self.done[i]:
                    i += 1
                    continue
                j = i
                while j < b and not self.done[j]:
                    j += 1
                out.append((i, j))
                i = j
        return out


class SourceBeauty:
    """한 원본의 설정 · 떨림을 줄인 얼굴 점 · 흐릴 얼굴 상자."""

    def __init__(self, settings: BeautySettings, track: np.ndarray, hide: dict | None) -> None:
        self.settings = settings
        self.track = track  # (프레임, 106, 2) smooth_track을 거친 것
        self.hide = hide or {}

    def landmarks(self, index: int) -> np.ndarray | None:
        if 0 <= index < len(self.track) and not np.isnan(self.track[index]).any():
            return self.track[index]
        return None


class TimelineBeauty:
    """내보내기 한 번 동안: 원본별 보정 준비(prepare)와 프레임마다 입히기(apply)."""

    def __init__(self, project: Project, workspace: Workspace, sources: SourceCache,
                 settings: dict[str, BeautySettings] | None = None) -> None:
        self._project, self._workspace, self._sources = project, workspace, sources
        used = {p.clip.source for p in project.placed()}
        chosen = active_settings(workspace) if settings is None else settings
        self.settings = {s: v for s, v in chosen.items() if s in used}
        self._ready: dict[str, SourceBeauty] = {}
        self._bodies: dict[str, object] = {}  # 클립 id → BodyFrames (앞 프레임 배경을 기억 — 클립마다 따로)

    @property
    def active(self) -> bool:
        return bool(self.settings)

    def _paths(self, source: str, settings: BeautySettings) -> tuple[Path, Path]:
        from ks_studio.services.beautify import track_path_for

        media = self._sources.media(self._workspace.resolve(source))
        return media, track_path_for(self._workspace.cache_dir, media, settings)

    def analysis_needed(self) -> int:
        """분석해야 할 프레임 수 (0이면 바로 내보낸다) — 진행률 나누기용."""
        total = 0
        for source, ranges in needed_ranges(self._project, set(self.settings)).items():
            media, track_path = self._paths(source, self.settings[source])
            if track_path.exists() or not media.exists():
                continue
            spans = spans_file(track_path)
            track = SpanTrack.load(spans, frame_count(media)) if spans.exists() else SpanTrack(frame_count(media))
            total += sum(b - a for a, b in track.missing(ranges))
        return total

    def prepare(self, on_progress=None, cancel=None) -> None:
        """원본마다 얼굴 점을 준비한다 (없는 구간만 분석해 캐시에 더한다). on_progress(0~1)."""
        todo = self.analysis_needed()
        done = 0
        detector = None
        for source, ranges in needed_ranges(self._project, set(self.settings)).items():
            settings = self.settings[source]
            media, track_path = self._paths(source, settings)
            if not media.exists():
                continue
            from ks_studio.services.beautify import face_blur_plan
            from ks_studio.services.face import analyze_video, everyone_file, load_everyone

            if track_path.exists():  # AI 보정 탭에서 영상 전체를 분석해 두었다
                landmarks = np.load(track_path).astype(np.float64)
                everyone_path = everyone_file(track_path.parent, cache_name(media))
                everyone = load_everyone(everyone_path) if settings.blur_others and everyone_path.exists() else []
            else:
                spans = spans_file(track_path)
                track = SpanTrack.load(spans, frame_count(media)) if spans.exists() else SpanTrack(frame_count(media))
                missing = track.missing(ranges)
                if missing:
                    if detector is None:
                        from ks_studio.services.face import FaceDetector

                        detector = FaceDetector()
                    for a, b in missing:
                        if cancel is not None and cancel():
                            track.save(spans)  # 한 구간까지는 남긴다
                            return
                        found: list = []
                        base = done

                        def report(r: float, base=base, size=b - a) -> None:
                            if on_progress and todo:
                                on_progress(min(1.0, (base + r * size) / todo))

                        rows = analyze_video(detector, media, settings.target_vector, report, start=a, count=b - a, everyone=found)
                        n = min(len(rows), len(track.done) - a)
                        track.landmarks[a : a + n] = rows[:n]
                        track.done[a : a + n] = True
                        track.everyone += found
                        done += b - a
                    track.save(spans)
                landmarks = track.landmarks.astype(np.float64)
                everyone = track.everyone
            smoothed = smooth_track(landmarks)
            hide = face_blur_plan(settings, everyone, smoothed) if settings.blur_others else None
            self._ready[source] = SourceBeauty(settings, smoothed, hide)
        if on_progress:
            on_progress(1.0)

    def apply(self, decoded: av.VideoFrame, clip: Clip, size: tuple[int, int]) -> av.VideoFrame:
        """원본 프레임에 그 원본의 보정을 입혀 size 크기(yuv420p)로. 설정이 없으면 그대로."""
        beauty = self._ready.get(clip.source)
        if beauty is None or decoded.time is None:
            return decoded
        from ks_studio.services.beautify import apply_frame, body_frames
        from ks_studio.services.blur import blur_frame

        index = round(decoded.time * clip.fps)
        w, h = size
        sx, sy = w / decoded.width, h / decoded.height
        frame = convert(decoded, "yuv420p", w, h)
        frame.time_base = decoded.time_base
        frame.pts = decoded.pts
        landmarks = beauty.landmarks(index)
        if landmarks is not None:
            landmarks = landmarks * (sx, sy)
        if clip.id not in self._bodies:
            self._bodies[clip.id] = body_frames(beauty.settings)
        bodies = self._bodies[clip.id]
        out = apply_frame(frame, landmarks, beauty.settings, bodies(frame, landmarks) if bodies else None)
        boxes = beauty.hide.get(index, ())
        if boxes:
            out = blur_frame(out, [(x0 * sx, y0 * sy, x1 * sx, y1 * sy) for x0, y0, x1, y1 in boxes], beauty.settings.blur_style)
        out.pts, out.time_base = decoded.pts, decoded.time_base
        return out


def work_size(source: tuple[int, int], output: tuple[int, int], clip: Clip) -> tuple[int, int]:
    """보정할 크기: 출력을 채우는 데 필요한 만큼만 (원본보다 크게는 안 함). 클립을 키웠거나 잘라 냈으면 그만큼 더 크게."""
    w, h = source
    need = max(output[0] / w, output[1] / h) * max(1.0, clip.scale)
    left, top, right, bottom = clip.crop
    need /= max(0.25, min(1 - left - right, 1 - top - bottom))
    scale = min(1.0, need)
    return max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
