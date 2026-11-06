"""자막 탭 플레이어: 영상 위에 지금 시각의 자막을 그린다 (내보낼 때 굽는 것과 같은 그림 — services.captions).

QVideoWidget은 영상을 따로 그려서 위에 글을 얹을 수 없다 → 프레임을 받아(QVideoSink) 직접 그린다.
"""

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ks_studio.core.subtitles import Subtitles
from ks_studio.services.captions import paint_caption
from ks_studio.ui.widgets.source_row import clock


class _Canvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image: QImage | None = None
        self.caption: str = ""
        self.subs = Subtitles()
        self.setMinimumSize(320, 180)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.image is None or self.image.isNull():
            return
        w, h = self.image.width(), self.image.height()
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
        self.canvas = _Canvas()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(44)
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.time = QLabel("0:00 / 0:00")
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda ms: self.slider.setRange(0, ms))
        self.player.playbackStateChanged.connect(
            lambda state: self.play_button.setText("⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")
        )
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

    def _on_frame(self, frame) -> None:
        image = frame.toImage()
        if not image.isNull():
            self.canvas.image = image
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
