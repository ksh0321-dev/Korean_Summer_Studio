"""완성된 영상 파일 하나를 트는 플레이어 (AI 보정 미리보기 영상, 내보내기 저해상도 미리보기).

편집 미리보기(playback.py)와 달리 이미 렌더한 파일을 그대로 튼다 — Qt Multimedia(QMediaPlayer).
Windows는 열려 있는 파일을 덮어쓸 수 없으므로, 같은 파일을 다시 렌더하기 전에 release()로 놓아 준다.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ks_studio.services.foreground import set_playing


def _clock(ms: int) -> str:
    seconds = max(0, ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


class ClipPlayer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.player.setVideoOutput(self.video)
        self.play_button = QPushButton("⏸")
        self.play_button.setFixedWidth(44)
        self.play_button.clicked.connect(self.toggle)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.sliderMoved.connect(self.player.setPosition)
        self.time = QLabel("0:00 / 0:00")
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda ms: self.position.setRange(0, ms))
        self.player.playbackStateChanged.connect(self._on_state)
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.position, 1)
        controls.addWidget(self.time)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video, 1)
        layout.addLayout(controls)

    def open(self, path: Path, play: bool = True) -> None:
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        if play:
            self.player.play()
        else:
            self.player.pause()  # 멈춘 채 첫 화면을 띄운다

    def release(self) -> None:
        """재생을 멈추고 파일을 놓는다 (다시 렌더해서 덮어쓰기 전에)."""
        self.player.stop()
        self.player.setSource(QUrl())

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_position(self, ms: int) -> None:
        if not self.position.isSliderDown():
            self.position.setValue(ms)
        self.time.setText(f"{_clock(ms)} / {_clock(self.player.duration())}")

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸" if playing else "▶")
        set_playing(f"clip-{id(self)}", playing)  # 재생하는 동안 백그라운드 작업을 쉬게
