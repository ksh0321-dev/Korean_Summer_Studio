"""타임라인 → 프레임 렌더. 미리보기·임시 내보내기·(나중에) AI 보정이 같이 쓴다.

타임라인 프레임 하나마다 (프레임 번호, 영상 프레임, 그 프레임 동안의 오디오 1/fps초)를 낸다.
영상과 오디오가 한 묶음으로 나오므로, 받는 쪽은 둘을 따로 맞출 필요가 없다.

영상: Project.visible_segments()가 정한 바탕(화면을 다 덮는 맨 위 클립) 위에 겹치는 클립들을 차례로 섞어 그린다.
오디오: 그 시각에 걸린 모든 줄(소리를 끈 줄 제외)의 클립 소리를 더한다. 화면에서 덮인 클립도 소리는 난다.
원본 대신 편집용 사본이 있으면 사본을 읽는다. 화면 비율이 다른 클립(세로 영상 등)은 늘리지 않고 검은 여백을 둔다.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from ks_studio.core.project import MAIN_TRACK, Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.sources import AUDIO_RATE, SourceCache, fit_rgb

log = logging.getLogger(__name__)


DEFAULT_CANVAS = (1920, 1080)


def canvas_size(project: Project, workspace: Workspace, sources: SourceCache) -> tuple[int, int]:
    """결과물 크기 = V1 첫 클립(없으면 처음 놓인 클립)의 크기. 마무리 단계에서 고를 수 있게 되기 전까지의 규칙."""
    placed = sorted(project.placed(), key=lambda p: (p.track, p.start))
    if placed:
        info = sources.info(workspace.resolve(placed[0].clip.source))
        if info is not None:
            return info.width, info.height
    return DEFAULT_CANVAS


def fit_size(size: tuple[int, int], box: tuple[int, int]) -> tuple[int, int]:
    """size의 비율을 지켜 box 안에 들어가는 크기 (짝수)."""
    scale = min(box[0] / size[0], box[1] / size[1])
    return max(2, round(size[0] * scale) // 2 * 2), max(2, round(size[1] * scale) // 2 * 2)


def layer_rect(width: int, height: int, canvas: tuple[int, int], clip: Clip) -> tuple[int, int, int, int]:
    """원본 (width, height)인 clip이 캔버스에 그려지는 영역 (x, y, w, h): 화면에 맞춘 크기 × scale, 가운데 = (pos_x, pos_y)."""
    fw, fh = fit_size((width, height), canvas)
    w, h = max(2, round(fw * clip.scale) // 2 * 2), max(2, round(fh * clip.scale) // 2 * 2)
    return round(clip.pos_x * canvas[0] - w / 2), round(clip.pos_y * canvas[1] - h / 2), w, h


class TimelineRenderer:
    def __init__(
        self,
        project: Project,
        workspace: Workspace,
        sources: SourceCache,
        size: tuple[int, int],
        pix_fmt: str,
    ) -> None:
        self._project = project
        self._workspace = workspace
        self._sources = sources
        self._size = size
        self._pix_fmt = pix_fmt
        self.samples_per_frame = AUDIO_RATE // project.fps

    def frames(self, start: int = 0, with_audio: bool = True) -> Iterator[tuple[int, av.VideoFrame, np.ndarray | None]]:
        audible = [p for p in self._project.placed() if p.track not in self._project.muted]
        overlaps = self._project.main_overlaps()
        audio_cache: dict[Path, np.ndarray | None] = {}

        def audio_at(t: int) -> np.ndarray | None:
            if not with_audio:
                return None
            mix = np.zeros(self.samples_per_frame, np.float32)
            for placed in audible:
                if placed.start <= t < placed.end:
                    path = self._workspace.resolve(placed.clip.source)
                    if path not in audio_cache:
                        audio_cache[path] = self._sources.audio(path) if path.exists() else None
                    crossfade = overlaps.get(placed.clip.id, (0, 0)) if placed.track == MAIN_TRACK else (0, 0)
                    mix += self._audio(audio_cache[path], placed.clip, t - placed.start, crossfade)
            # 여러 소리를 더하면 1.0을 넘을 수 있다 → 넘는 부분만 잘라 찢어지는 소리를 막는다
            return np.clip(mix, -1.0, 1.0, out=mix)

        # 원본마다 디코더 하나를 열어 두고 차례로 프레임을 넘긴다 (원본 fps가 높으면 건너뛰고, 낮으면 같은 프레임을 다시 쓴다).
        # 바탕(화면을 덮는 클립)과 겹치는 클립은 변환이 달라서 따로 연다.
        decoders: dict[tuple, list] = {}

        def frame_of(clip: Clip, clip_start: int, t: int, as_layer: bool):
            key = (clip.id, clip_start, as_layer)
            want = self._project.source_frame(clip, t - clip_start)
            if key not in decoders:
                path = self._sources.media(self._workspace.resolve(clip.source))
                convert = (lambda d, c=clip: self._layer_image(d, c)) if as_layer else self._fit
                blank = (lambda c=clip: self._blank_layer(c)) if as_layer else self._black
                source = self._video(path, clip.fps, want, clip.out_frame, convert, blank)
                decoders[key] = [source, *next(source)]
            entry = decoders[key]
            while entry[1] < want:
                entry[1], entry[2] = next(entry[0])
            return entry[2]

        try:
            for segment in self._project.visible_segments():
                if segment.end <= start:
                    continue
                needed = {(layer.clip.id, layer.clip_start, True) for layer in segment.above}
                if segment.clip is not None:
                    needed.add((segment.clip.id, segment.clip_start, False))
                for key in [k for k in decoders if k not in needed]:  # 다 쓴 원본은 바로 닫는다
                    decoders.pop(key)[0].close()
                for t in range(max(start, segment.start), segment.end):
                    base = self._black() if segment.clip is None else frame_of(segment.clip, segment.clip_start, t, False)
                    if not segment.above:
                        yield t, base, audio_at(t)
                        continue
                    canvas = base.to_ndarray(format="rgb24").copy()
                    for layer in segment.above:
                        image = frame_of(layer.clip, layer.clip_start, t, True)
                        alpha = layer.clip.opacity
                        if layer.dissolve_frames:  # 디졸브: 겹친 동안 0 → 1로 서서히
                            alpha *= (t - layer.clip_start + 0.5) / layer.dissolve_frames
                        self._paste(canvas, image, layer.clip, min(1.0, alpha))
                    frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                    yield t, frame if self._pix_fmt == "rgb24" else frame.reformat(format=self._pix_fmt), audio_at(t)
        finally:
            for entry in decoders.values():
                entry[0].close()

    def _video(self, path: Path, fps: float, first: int, last: int, convert, blank) -> Iterator[tuple[int, object]]:
        """원본 프레임 [first, last)를 하나씩 정확히 한 번 낸다 (convert로 바꿔서). 빠진 프레임은 직전 것으로, 못 읽으면 blank()."""
        produced, previous = first, None
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                container.seek(int(first / (fps * stream.time_base)), stream=stream, backward=True)
                for decoded in container.decode(stream):
                    k = round(decoded.pts * stream.time_base * fps)
                    if k < produced:
                        continue
                    frame = convert(decoded)
                    while produced < min(k, last):
                        yield produced, frame if previous is None else previous
                        produced += 1
                    if produced >= last:
                        return
                    yield produced, frame
                    produced += 1
                    previous = frame
                    if produced >= last:
                        return
        except (av.FFmpegError, OSError, IndexError) as e:
            log.warning("원본을 읽을 수 없음 %s: %s", path, e)
        filler = blank() if previous is None else previous
        for k in range(produced, last):
            yield k, filler

    def _layer_image(self, decoded: av.VideoFrame, clip: Clip) -> np.ndarray:
        """겹쳐 그릴 RGB 그림. 화면을 덮는 클립(디졸브로 나타나는 중)은 바탕과 똑같이 검은 여백까지 맞추고,
        작게·옆으로·반투명한 클립은 여백 없이 그 크기로만 (여백이 아래 줄을 가리지 않게)."""
        if clip.covers:
            return fit_rgb(decoded, *self._size)
        _, _, w, h = layer_rect(decoded.width, decoded.height, self._size, clip)
        return decoded.reformat(width=w, height=h, format="rgb24").to_ndarray()

    def _blank_layer(self, clip: Clip) -> np.ndarray:
        return np.zeros((2, 2, 3), np.uint8)

    def _paste(self, canvas: np.ndarray, image: np.ndarray, clip: Clip, alpha: float) -> None:
        """image를 clip의 위치에 alpha만큼 섞어 그린다 (화면 밖으로 나간 부분은 자른다)."""
        h, w = image.shape[:2]
        H, W = canvas.shape[:2]
        if (w, h) == (W, H) and clip.pos_x == 0.5 and clip.pos_y == 0.5:
            x0, y0 = 0, 0
        else:
            x0, y0 = round(clip.pos_x * W - w / 2), round(clip.pos_y * H - h / 2)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if cx0 >= cx1 or cy0 >= cy1 or alpha <= 0:
            return
        part = image[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        if alpha >= 1:
            canvas[cy0:cy1, cx0:cx1] = part
        else:
            region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
            canvas[cy0:cy1, cx0:cx1] = (region + (part.astype(np.float32) - region) * alpha).astype(np.uint8)

    def _audio(self, audio: np.ndarray | None, clip: Clip, local: int, crossfade: tuple[int, int] = (0, 0)) -> np.ndarray:
        """클립 안 타임라인 위치 local 동안의 오디오. 오프셋이 +면 소리를 늦추므로 원본에서 그만큼 앞을 가져온다."""
        n = self.samples_per_frame
        out = np.zeros(n, np.float32)
        if audio is None:
            return out
        start = round(self._project.source_time(clip, local) * AUDIO_RATE) - round(clip.audio_offset_ms * AUDIO_RATE / 1000)
        lo, hi = max(0, start), min(len(audio), start + n)
        if lo < hi:
            out[lo - start : hi - start] = audio[lo:hi] / 32768
        if clip.volume_db or clip.fade_in_ms or clip.fade_out_ms or any(crossfade):
            out *= self._gain(clip, local, n, crossfade)
        return out

    def _gain(self, clip: Clip, local: int, n: int, crossfade: tuple[int, int] = (0, 0)) -> np.ndarray:
        """볼륨(dB) × 페이드 곡선 (직선). 시각은 클립이 타임라인에서 시작한 때부터.

        crossfade = V1 디졸브로 앞·뒤 클립과 겹친 프레임 수 → 그동안 소리도 교차한다 (앞 클립은 줄고 뒤 클립은 커진다).
        """
        fps = self._project.fps
        t = local / fps + np.arange(n) / AUDIO_RATE
        length = self._project.length(clip) / fps
        envelope = np.ones(n, np.float32)
        fade_in = max(clip.fade_in_ms / 1000, crossfade[0] / fps)
        fade_out = max(clip.fade_out_ms / 1000, crossfade[1] / fps)
        if fade_in:
            envelope = np.minimum(envelope, t / fade_in)
        if fade_out:
            envelope = np.minimum(envelope, (length - t) / fade_out)
        return np.clip(envelope, 0, 1) * 10 ** (clip.volume_db / 20)

    def _fit(self, decoded: av.VideoFrame) -> av.VideoFrame:
        """출력 크기에 맞춘다. 비율이 같으면 그대로 늘이고, 다르면 비율을 지켜 가운데에 놓고 검은 여백을 둔다."""
        w, h = self._size
        if decoded.width * h == decoded.height * w:
            return decoded.reformat(width=w, height=h, format=self._pix_fmt)
        frame = av.VideoFrame.from_ndarray(fit_rgb(decoded, w, h), format="rgb24")
        return frame if self._pix_fmt == "rgb24" else frame.reformat(format=self._pix_fmt)

    def _black(self) -> av.VideoFrame:
        w, h = self._size
        return av.VideoFrame.from_ndarray(np.zeros((h, w, 3), np.uint8), format="rgb24").reformat(format=self._pix_fmt)
