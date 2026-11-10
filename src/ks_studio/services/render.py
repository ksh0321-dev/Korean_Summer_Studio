"""타임라인 → 프레임 렌더. 미리보기·임시 내보내기·(나중에) AI 보정이 같이 쓴다.

타임라인 프레임 하나마다 (프레임 번호, 영상 프레임, 그 프레임 동안의 오디오 1/fps초)를 낸다.
영상과 오디오가 한 묶음으로 나오므로, 받는 쪽은 둘을 따로 맞출 필요가 없다.

영상: Project.visible_segments()가 정한 바탕(화면을 다 덮는 맨 위 클립) 위에 겹치는 클립들을 차례로 섞어 그린다.
오디오: 그 시각에 걸린 모든 줄(소리를 끈 줄 제외)의 클립 소리를 더한다. 화면에서 덮인 클립도 소리는 난다.
       소리 줄(배경 음악·떼어 낸 소리)도 더한다 — 배경 음악은 말소리가 나는 동안 저절로 작아진다 (덕킹).
원본 대신 편집용 사본이 있으면 사본을 읽는다. 화면 비율이 다른 클립(세로 영상 등)은 늘리지 않고 검은 여백을 둔다.
클립마다 자르기·회전·뒤집기·색 보정·LUT를 맞추기 전에 입히고, 속도를 바꾼 클립 소리는 높이를 지킨 채 늘이고 줄인다 (atempo).
"""

import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import av.filter
import numpy as np

from ks_studio.core.color import grade, shape
from ks_studio.core.project import MAIN_TRACK, Clip, Project, Sound
from ks_studio.core.workspace import Workspace
from ks_studio.services.yuv import convert, from_rgb, to_rgb
from ks_studio.services.sources import AUDIO_RATE, SourceCache, fit_rgb

log = logging.getLogger(__name__)


DEFAULT_CANVAS = (1920, 1080)
DUCK_GAIN = 0.25  # 말소리가 나는 동안 배경 음악 크기 (-12dB)
VOICE_LEVEL = 0.02  # 클립 소리(RMS)가 이보다 크면 말소리가 난다고 본다 (약 -34dBFS)
DUCK_ATTACK, DUCK_RELEASE = 0.5, 0.08  # 프레임마다 목표 크기 쪽으로 가는 비율: 줄 때는 빨리, 돌아올 때는 천천히


def stretch_audio(audio: np.ndarray, speed: float) -> np.ndarray:
    """48kHz 모노 int16을 speed배 빠르게 (높이는 그대로 — FFmpeg atempo, 한 번에 0.5~2배라 여러 번 잇는다)."""
    if speed == 1.0 or not len(audio):
        return audio
    graph = av.filter.Graph()
    nodes = [graph.add_abuffer(format="s16", sample_rate=AUDIO_RATE, layout="mono", time_base=Fraction(1, AUDIO_RATE))]
    left = speed
    while left > 2.0:
        nodes.append(graph.add("atempo", "2.0"))
        left /= 2.0
    while left < 0.5:
        nodes.append(graph.add("atempo", "0.5"))
        left /= 0.5
    nodes.append(graph.add("atempo", f"{left:.6f}"))
    nodes.append(graph.add("abuffersink"))
    graph.link_nodes(*nodes).configure()
    out = []

    def drain() -> None:
        while True:
            try:
                out.append(graph.pull().to_ndarray().reshape(-1))
            except (av.BlockingIOError, av.EOFError):
                return

    chunk = AUDIO_RATE // 10
    for i in range(0, len(audio), chunk):
        part = np.ascontiguousarray(audio[i : i + chunk])
        frame = av.AudioFrame.from_ndarray(part.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
        graph.push(frame)
        drain()
    graph.push(None)
    drain()
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, np.int16)


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
        beauty=None,
    ) -> None:
        """beauty = services.timeline_beauty.TimelineBeauty (내보낼 때만 — 준비(prepare)를 마친 것). 원본 프레임마다 AI 보정."""
        self._beauty = beauty
        self._project = project
        self._workspace = workspace
        self._sources = sources
        self._size = size
        self._pix_fmt = pix_fmt
        self.samples_per_frame = AUDIO_RATE // project.fps
        self._luts: dict[tuple, av.filter.Graph] = {}
        self._cubes: dict[str, Path | None] = {}  # 원본 → 색 보정 .cube (색 보정 탭)

    def audio_track(self) -> np.ndarray:
        """타임라인 전체 소리 (48kHz 모노 float32, 영상은 풀지 않는다) — 자막 음성 인식용."""
        audio_at = self._audio_mixer(True)
        mixed = [audio_at(t) for t in range(self._project.duration)]
        return np.concatenate(mixed) if mixed else np.zeros(0, np.float32)

    def _audio_mixer(self, with_audio: bool):
        """프레임 t의 소리를 섞어 주는 함수 (frames와 audio_track이 같이 쓴다). 차례로 부르면 덕킹이 이어진다."""
        audible = [p for p in self._project.placed() if p.track not in self._project.muted and not p.clip.detached]
        overlaps = self._project.main_overlaps()
        fps = self._project.fps
        sounds = [(s, s.start, s.start + round(s.length_ms * fps / 1000)) for s in self._project.sounds]
        audio_cache: dict[tuple, np.ndarray | None] = {}
        duck = [1.0]  # 배경 음악 크기 (프레임마다 목표 쪽으로 서서히)

        def source_audio(source: str, speed: float = 1.0) -> np.ndarray | None:
            path = self._workspace.resolve(source)
            key = (path, speed)
            if key not in audio_cache:
                if (path, 1.0) not in audio_cache:
                    audio_cache[(path, 1.0)] = self._sources.audio(path) if path.exists() else None
                plain = audio_cache[(path, 1.0)]
                audio_cache[key] = stretch_audio(plain, speed) if plain is not None and speed != 1.0 else plain
            return audio_cache[key]

        def audio_at(t: int) -> np.ndarray | None:
            if not with_audio:
                return None
            mix = np.zeros(self.samples_per_frame, np.float32)
            for placed in audible:
                if placed.start <= t < placed.end:
                    crossfade = overlaps.get(placed.clip.id, (0, 0)) if placed.track == MAIN_TRACK else (0, 0)
                    audio = source_audio(placed.clip.source, placed.clip.speed)
                    mix += self._audio(audio, placed.clip, t - placed.start, crossfade)
            if sounds:
                voice = float(np.sqrt(np.mean(mix * mix))) if mix.any() else 0.0
                target = DUCK_GAIN if voice > VOICE_LEVEL else 1.0
                duck[0] += (target - duck[0]) * (DUCK_ATTACK if target < duck[0] else DUCK_RELEASE)
                for sound, start, end in sounds:
                    if start <= t < end:
                        part = self._sound(source_audio(sound.source), sound, t - start, end - start)
                        mix += part * duck[0] if sound.duck else part
            # 여러 소리를 더하면 1.0을 넘을 수 있다 → 넘는 부분만 잘라 찢어지는 소리를 막는다
            return np.clip(mix, -1.0, 1.0, out=mix)

        return audio_at

    def _sound(self, audio: np.ndarray | None, sound: Sound, local: int, length: int) -> np.ndarray:
        """소리 줄 조각의 프레임 local 동안의 소리 (볼륨·페이드 포함)."""
        n = self.samples_per_frame
        out = np.zeros(n, np.float32)
        if audio is None:
            return out
        fps = self._project.fps
        start = round(sound.in_ms * AUDIO_RATE / 1000) + local * n
        lo, hi = max(0, start), min(len(audio), start + n)
        if lo < hi:
            out[lo - start : hi - start] = audio[lo:hi] / 32768
        t = local / fps + np.arange(n) / AUDIO_RATE
        envelope = np.ones(n, np.float32)
        if sound.fade_in_ms:
            envelope = np.minimum(envelope, t / (sound.fade_in_ms / 1000))
        if sound.fade_out_ms:
            envelope = np.minimum(envelope, (length / fps - t) / (sound.fade_out_ms / 1000))
        return out * np.clip(envelope, 0, 1) * 10 ** (sound.volume_db / 20)

    def frames(self, start: int = 0, with_audio: bool = True) -> Iterator[tuple[int, av.VideoFrame, np.ndarray | None]]:
        audio_at = self._audio_mixer(with_audio)

        # 원본마다 디코더 하나를 열어 두고 차례로 프레임을 넘긴다 (원본 fps가 높으면 건너뛰고, 낮으면 같은 프레임을 다시 쓴다).
        # 바탕(화면을 덮는 클립)과 겹치는 클립은 변환이 달라서 따로 연다.
        decoders: dict[tuple, list] = {}

        def frame_of(clip: Clip, clip_start: int, t: int, as_layer: bool):
            key = (clip.id, clip_start, as_layer)
            want = self._project.source_frame(clip, t - clip_start)
            if key not in decoders:
                path = self._sources.media(self._workspace.resolve(clip.source))
                if as_layer:
                    convert = lambda d, c=clip: self._layer_image(self._prepare(d, c), c)  # noqa: E731
                else:
                    convert = lambda d, c=clip: self._fit(self._prepare(d, c))  # noqa: E731
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
                    canvas = to_rgb(base).copy()
                    for layer in segment.above:
                        image = frame_of(layer.clip, layer.clip_start, t, True)
                        alpha = layer.clip.opacity
                        if layer.dissolve_frames:  # 디졸브: 겹친 동안 0 → 1로 서서히
                            alpha *= (t - layer.clip_start + 0.5) / layer.dissolve_frames
                        self._paste(canvas, image, layer.clip, min(1.0, alpha))
                    frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                    yield t, convert(frame, self._pix_fmt), audio_at(t)
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

    def _source_cube(self, source: str) -> Path | None:
        if source not in self._cubes:
            from ks_studio.services.grade import source_cube

            self._cubes[source] = source_cube(self._workspace, source)
        return self._cubes[source]

    def _prepare(self, decoded: av.VideoFrame, clip: Clip) -> av.VideoFrame:
        """맞추기 전에: (내보낼 때) AI 보정 → 원본 색 보정(색 보정 탭) → 클립 LUT → 자르기·회전·뒤집기 → 클립 색.
        아무것도 안 했으면 그대로 (변환 비용 없음). AI 보정은 출력을 채울 만큼 줄인 크기로 입힌다 (services.timeline_beauty)."""
        if self._beauty is not None and clip.source in self._beauty.settings:
            from ks_studio.services.timeline_beauty import work_size

            decoded = self._beauty.apply(decoded, clip, work_size((decoded.width, decoded.height), self._size, clip))
        cube = self._source_cube(clip.source)
        if not (clip.shaped or clip.graded or cube):
            return decoded
        rgb = to_rgb(decoded)
        if cube is not None:
            rgb = self._apply_lut(rgb, str(cube))
        if not (clip.shaped or clip.graded):
            return av.VideoFrame.from_ndarray(rgb, format="rgb24")
        if clip.lut:
            rgb = self._apply_lut(rgb, clip.lut)
        if clip.shaped:
            rgb = shape(rgb, clip.crop, clip.rotation, clip.flip)
        rgb = grade(rgb, clip.brightness, clip.contrast, clip.saturation, clip.temperature, clip.look)
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")

    def _apply_lut(self, rgb: np.ndarray, lut: str) -> np.ndarray:
        """.cube LUT (FFmpeg lut3d). 파일을 못 읽으면 그대로 (경고만)."""
        h, w = rgb.shape[:2]
        key = (lut, w, h)
        try:
            if key not in self._luts:
                graph = av.filter.Graph()
                source = graph.add_buffer(width=w, height=h, format="rgb24", time_base=Fraction(1, 30))
                # 경로는 이름 붙인 옵션으로 (필터 글로 쓰면 C: 같은 콜론을 따로 막아야 한다)
                graph.link_nodes(source, graph.add("lut3d", file=str(self._workspace.resolve(lut))), graph.add("buffersink")).configure()
                self._luts[key] = graph
            graph = self._luts[key]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = 0
            graph.push(frame)
            return graph.pull().to_ndarray(format="rgb24")
        except (av.FFmpegError, ValueError) as e:
            log.warning("LUT을 쓸 수 없음 %s: %s", lut, e)
            self._luts.pop(key, None)
            return rgb

    def _layer_image(self, decoded: av.VideoFrame, clip: Clip) -> np.ndarray:
        """겹쳐 그릴 RGB 그림. 화면을 덮는 클립(디졸브로 나타나는 중)은 바탕과 똑같이 검은 여백까지 맞추고,
        작게·옆으로·반투명한 클립은 여백 없이 그 크기로만 (여백이 아래 줄을 가리지 않게)."""
        if clip.covers:
            return fit_rgb(decoded, *self._size)
        _, _, w, h = layer_rect(decoded.width, decoded.height, self._size, clip)
        return to_rgb(decoded, w, h)

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
        # 속도를 바꾼 클립의 audio는 이미 speed배로 늘이고 줄인 것 → 원본 시각을 speed로 나눈 자리
        heard = self._project.source_time(clip, local) - clip.audio_offset_ms / 1000
        start = round(heard / clip.speed * AUDIO_RATE)
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
            return convert(decoded, self._pix_fmt, w, h)
        return from_rgb(fit_rgb(decoded, w, h), self._pix_fmt)

    def _black(self) -> av.VideoFrame:
        w, h = self._size
        return from_rgb(np.zeros((h, w, 3), np.uint8), self._pix_fmt)
