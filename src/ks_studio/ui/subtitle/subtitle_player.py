"""자막 탭 플레이어: 영상 위에 지금 시각의 자막을 그린다 (내보낼 때 굽는 것과 같은 그림 — services.captions).

QVideoWidget은 영상을 따로 그려서 위에 글을 얹을 수 없다 → 프레임을 받아(QVideoSink) 직접 그린다.
프레임을 그림으로 바꾸고 화면 크기로 줄이는 일은 작업 스레드에서 한다 — UI 스레드에서 하면 4K 60fps가
46fps까지만 나오고 버튼 반응도 느려졌다 (실측: 4K 프레임 하나 toImage 15ms). 한 스레드로는 변환+줄이기가 20ms라
47fps였다 → 스레드 둘이 번갈아 맡는다. 둘 다 바쁘면 그 프레임은 건너뛴다 (밀리지 않게).
"""

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ks_studio.core.subtitles import Subtitles
from ks_studio.services.captions import paint_caption
from ks_studio.services.foreground import set_playing
from ks_studio.ui.widgets.latest_worker import LatestWorker
from ks_studio.ui.widgets.source_row import clock


def _shrink(frame, box: QSize):
    """(화면 크기로 줄인 그림, 원래 가로, 원래 세로) — 작업 스레드에서. 자막 자리는 원래 크기 기준으로 잡는다."""
    image = frame.toImage()
    if image.isNull():
        return None
    w, h = image.width(), image.height()
    if box.width() > 0 and box.height() > 0 and (w > box.width() or h > box.height()):
        image = image.scaled(box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return image, w, h


CONVERTERS = 2  # 프레임 변환 스레드 수


class _Canvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image: QImage | None = None
        self.frame_size = (0, 0)  # 원래 영상 크기 (image는 화면 크기로 줄인 것)
        self.caption: str = ""
        self.subs = Subtitles()
        self.setMinimumSize(320, 180)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.image is None or self.image.isNull():
            return
        w, h = self.frame_size
        scale = min(self.width() / w, self.height() / h)
        target = QRectF((self.width() - w * scale) / 2, (self.height() - h * scale) / 2, w * scale, h * scale)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self.image)
        if self.caption:
            paint_caption(painter, target, self.caption, self.subs.style, w, h)


class SubtitlePlayer(QWidget):
    position_changed = Signal(float)  # 초

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)
        self._converters = [LatestWorker(f"subtitle-frames-{i}", self) for i in range(CONVERTERS)]
        for converter in self._converters:
            converter.ready.connect(self._on_converted)
        self._sequence = 0  # 프레임 순번 (두 스레드 결과가 순서가 바뀌어 와도 옛 프레임은 버린다)
        self._shown = 0
        self.canvas = _Canvas()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(44)
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.time = QLabel("0:00 / 0:00")
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda ms: self.slider.setRange(0, ms))
        self.player.playbackStateChanged.connect(self._on_state)
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)

    def open(self, path: Path) -> None:
        self.canvas.image = None
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.pause()  # 첫 화면을 띄운다

    def release(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self.canvas.image = None
        self.canvas.update()

    def set_subtitles(self, subs: Subtitles) -> None:
        self.canvas.subs = subs
        self._update_caption()

    def seek(self, seconds: float) -> None:
        self.player.setPosition(round(seconds * 1000))

    def seconds(self) -> float:
        return self.player.position() / 1000

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸" if playing else "▶")
        set_playing(f"subtitle-{id(self)}", playing)  # 재생하는 동안 백그라운드 작업을 쉬게

    def _on_frame(self, frame) -> None:
        idle = [c for c in self._converters if not c.busy]
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if not idle and playing:
            return  # 둘 다 바쁘다 → 이 프레임은 건너뛴다
        self._sequence += 1
        box = self.canvas.size() * self.canvas.devicePixelRatioF()
        (idle or self._converters)[0].request(lambda f=frame, b=box, n=self._sequence: (n, _shrink(f, b)))

    def _on_converted(self, _token: int, result) -> None:
        sequence, shrunk = result
        if shrunk is None or sequence < self._shown:
            return
        self._shown = sequence
        self.canvas.image, *size = shrunk
        self.canvas.frame_size = tuple(size)
        self.canvas.update()

    def _on_position(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time.setText(f"{clock(ms / 1000)} / {clock(self.player.duration() / 1000)}")
        self._update_caption()
        self.position_changed.emit(ms / 1000)

    def _update_caption(self) -> None:
        index = self.canvas.subs.at(self.seconds())
        text = self.canvas.subs.cues[index].text if index is not None else ""
        if text != self.canvas.caption:
            self.canvas.caption = text
            self.canvas.update()
