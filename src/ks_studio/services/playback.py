"""컷편집 미리보기 재생. 오디오를 기준 시계로 삼아, 지금 소리가 나는 위치의 프레임을 보여준다.

- 렌더 스레드: TimelineRenderer가 (프레임, 오디오) 묶음을 만들어 큐에 넣는다
- UI 타이머(10ms): 오디오 출력 버퍼가 비는 만큼 묶음을 꺼내 소리를 쓰고, 영상은 대기열에 둔다
  → 실제로 재생된 오디오 위치에 도달한 프레임을 화면에 낸다
- 멈춰 있을 때 찾아가기: 별도 스레드가 '가장 최근 요청'만 처리한다 (스크럽 중 밀린 요청은 버림)
"""

import logging
import queue
import threading
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.foreground import set_playing
from ks_studio.services.render import TimelineRenderer, canvas_size, fit_size
from ks_studio.services.sources import AUDIO_RATE, SourceCache

log = logging.getLogger(__name__)

PREVIEW_BOX = (960, 540)  # 미리보기는 결과물 비율을 지킨 채 이 안에 들어가는 크기로 그린다
QUEUE_FRAMES = 12
AUDIO_BUFFER_S = 0.2
BYTES_PER_SAMPLE = 4  # float32 모노


class Player(QObject):
    frame_ready = Signal(object)  # np.ndarray (H, W, 3) RGB
    position_changed = Signal(int)
    playing_changed = Signal(bool)

    def __init__(self, sources: SourceCache, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._project = Project()
        self._workspace: Workspace | None = None
        self.position = 0

        self._queue: queue.Queue | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: QAudioSink | None = None
        self._io = None
        self._shown: deque[tuple[int, np.ndarray]] = deque()
        self._start_frame = 0
        self._ended = False
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        # 윈도우 기본 타이머는 15.6ms 단위라 60fps(16.7ms)에서 두 프레임이 한 번에 몰려 하나가 버려진다
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        self._still_request: int | None = None
        self._still_event = threading.Event()
        threading.Thread(target=self._still_worker, daemon=True, name="still-frame").start()

    @property
    def playing(self) -> bool:
        return self._sink is not None

    def set_project(self, project: Project, workspace: Workspace) -> None:
        """편집하면 호출. 재생 중이면 멈추고 현재 위치의 정지 화면을 다시 그린다."""
        self.pause()
        self._project, self._workspace = project, workspace
        self.position = min(self.position, max(0, project.duration - 1))
        self.position_changed.emit(self.position)
        self._request_still()

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def play(self) -> None:
        if self.playing or self._workspace is None or self._project.duration == 0:
            return
        if self.position >= self._project.duration - 1:
            self.position = 0
        self._start_frame = self.position
        self._ended = False
        self._shown.clear()
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=QUEUE_FRAMES)
        renderer = self._renderer()
        self._thread = threading.Thread(
            target=self._produce, args=(renderer, self.position, self._queue, self._stop), daemon=True, name="playback"
        )
        self._thread.start()

        fmt = QAudioFormat()
        fmt.setSampleRate(AUDIO_RATE)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self._sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt, self)
        self._sink.setBufferSize(int(AUDIO_RATE * AUDIO_BUFFER_S) * BYTES_PER_SAMPLE)
        self._io = self._sink.start()
        self._timer.start()
        set_playing("editor", True)  # 썸네일·편집용 사본·AI 작업은 재생이 멈출 때까지 쉰다
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self.playing:
            return
        self._timer.stop()
        self._stop.set()
        self._sink.stop()
        self._sink.deleteLater()
        self._sink = self._io = None
        self._drain()
        set_playing("editor", False)
        self.playing_changed.emit(False)

    def seek(self, frame: int) -> None:
        was_playing = self.playing
        self.pause()
        self.position = max(0, min(frame, max(0, self._project.duration - 1)))
        self.position_changed.emit(self.position)
        if was_playing:
            self.play()
        else:
            self._request_still()

    def step(self, delta: int) -> None:
        self.seek(self.position + delta)

    # ── 재생 ────────────────────────────────────────────────────

    def _renderer(self) -> TimelineRenderer:
        size = fit_size(canvas_size(self._project, self._workspace, self._sources), PREVIEW_BOX)
        return TimelineRenderer(self._project, self._workspace, self._sources, size, "rgb24")

    @staticmethod
    def _produce(renderer: TimelineRenderer, start: int, q: queue.Queue, stop: threading.Event) -> None:
        try:
            for t, frame, audio in renderer.frames(start):
                item = (t, frame.to_ndarray(), audio)
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except Exception:
            log.exception("재생 렌더 실패")
        q.put(None)

    def _tick(self) -> None:
        samples = self._sink_samples_free()
        n = AUDIO_RATE // self._project.fps
        while not self._ended and samples >= n:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._ended = True
                break
            t, image, audio = item
            self._io.write(audio.tobytes())
            self._shown.append((t, image))
            samples -= n

        # processedUSecs = 출력 버퍼에서 장치로 넘어간 양 (실측: 쓴 양 - 버퍼에 남은 양) = 지금 재생 위치
        played_s = self._sink.processedUSecs() / 1_000_000
        now = self._start_frame + int(played_s * self._project.fps)
        latest = None
        while self._shown and self._shown[0][0] <= now:
            latest = self._shown.popleft()
        if latest is not None:
            self.position = latest[0]
            self.frame_ready.emit(latest[1])
            self.position_changed.emit(self.position)
        if self._ended and not self._shown and self._buffered_seconds() <= 0:
            self.pause()

    def _sink_samples_free(self) -> int:
        return self._sink.bytesFree() // BYTES_PER_SAMPLE

    def _buffered_seconds(self) -> float:
        """출력 버퍼에 쓰였지만 아직 재생되지 않은 오디오 길이."""
        return (self._sink.bufferSize() - self._sink.bytesFree()) / BYTES_PER_SAMPLE / AUDIO_RATE

    def _drain(self) -> None:
        self._shown.clear()
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    # ── 정지 화면 ────────────────────────────────────────────────

    def _request_still(self) -> None:
        if self._workspace is None or self._project.duration == 0:
            self.frame_ready.emit(None)
            return
        self._still_request = self.position
        self._still_event.set()

    def _still_worker(self) -> None:
        while True:
            self._still_event.wait()
            self._still_event.clear()
            target, renderer = self._still_request, self._renderer()
            try:
                item = next(renderer.frames(target, with_audio=False), None)
            except Exception:
                log.exception("정지 화면 렌더 실패")
                continue
            if item is not None and target == self._still_request and not self.playing:
                self.frame_ready.emit(item[1].to_ndarray())
