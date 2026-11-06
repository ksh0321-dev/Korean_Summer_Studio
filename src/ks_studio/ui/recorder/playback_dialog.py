from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class PlaybackDialog(QDialog):
    """테스트 녹화 결과를 바로 재생해 확인한다."""

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("테스트 녹화 확인")
        self.resize(960, 640)

        video = QVideoWidget()
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(QAudioOutput(self))
        self.player.setVideoOutput(video)
        self.player.setSource(QUrl.fromLocalFile(str(path)))

        hint = QLabel(
            "확인할 것: ① 화면 속 글자가 정상 방향인가  ② 박수·입 모양과 소리가 맞는가  ③ 소리가 깨지거나 너무 작지 않은가"
        )
        hint.setWordWrap(True)
        replay = QPushButton("다시 재생")
        replay.clicked.connect(self._replay)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(replay)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(video, 1)
        layout.addWidget(hint)
        layout.addLayout(buttons)
        self.player.play()

    def _replay(self) -> None:
        self.player.setPosition(0)
        self.player.play()

    def done(self, result: int) -> None:
        # 파일을 놓아줘야 다음 테스트 녹화가 덮어쓸 수 있다
        self.player.stop()
        self.player.setSource(QUrl())
        super().done(result)
