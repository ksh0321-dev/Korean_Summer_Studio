"""프로그램 정보 창: 버전 · 저장소 · 쓰는 AI 모델 · 로그 폴더 · 새 버전 확인."""

import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ks_studio import __version__
from ks_studio.app.paths import FROZEN, USER_DIR
from ks_studio.services.accel import open_folder
from ks_studio.services.updates import REPO_URL, check_for_update

MODELS = (  # (모델, 쓰는 곳)
    ("InsightFace buffalo_l", "얼굴 찾기 · 사람 알아보기"),
    ("Whisper large-v3 (faster-whisper / mlx-whisper)", "자동 자막"),
    ("MediaPipe Selfie Multiclass", "사람 분리 (몸매 · 얼굴 작게 · 배경 흐림)"),
    ("RTMPose (rtmlib)", "몸의 점 (몸매)"),
    ("MI-GAN · LaMa", "드러난 배경 메꾸기"),
)


class UpdateChecker(QObject):
    """새 버전 확인을 백그라운드 스레드에서. 끝나면 finished(새 버전 또는 "")."""

    finished = Signal(str)

    def start(self) -> None:
        threading.Thread(target=lambda: self.finished.emit(check_for_update() or ""), daemon=True).start()


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KS Studio 정보")
        models = "".join(f"<li><b>{name}</b> — {use}</li>" for name, use in MODELS)
        info = QLabel(
            f"<h2>KS Studio {__version__}</h2>"
            "<p>Korean Summer 채널용 녹화 · 컷편집 · AI 보정 · 내보내기 프로그램</p>"
            f"<p>파이썬 {sys.version.split()[0]} · {'배포판(exe)' if FROZEN else '소스 실행'}</p>"
            f'<p><a href="{REPO_URL}">{REPO_URL}</a></p>'
            f"<p>쓰는 AI 모델 (처음 쓸 때 내려받아 models/ 폴더에 둡니다)</p><ul>{models}</ul>"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        self.update_label = QLabel("")
        self.update_button = QPushButton("새 버전 확인")
        self.update_button.clicked.connect(self._check)
        logs = QPushButton("로그 폴더 열기")
        logs.setToolTip("문제가 생겼을 때 이 폴더의 ks_studio.log를 보내 주세요")
        logs.clicked.connect(lambda: open_folder(USER_DIR / "logs") if (USER_DIR / "logs").exists() else None)
        logs.setEnabled(FROZEN)  # 소스로 돌리면 로그는 터미널에
        row = QHBoxLayout()
        row.addWidget(self.update_button)
        row.addWidget(logs)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addLayout(row)
        layout.addWidget(self.update_label)
        layout.addWidget(buttons)
        self._checker = UpdateChecker(self)
        self._checker.finished.connect(self._show_update)

    def _check(self) -> None:
        self.update_button.setEnabled(False)
        self.update_label.setText("확인 중…")
        self._checker.start()

    def _show_update(self, version: str) -> None:
        self.update_button.setEnabled(True)
        if version:
            self.update_label.setText(f'새 버전 <b>{version}</b>이 있습니다 — <a href="{REPO_URL}">저장소에서 받기</a>')
            self.update_label.setOpenExternalLinks(True)
        else:
            self.update_label.setText("지금이 최신 버전입니다 (또는 인터넷·저장소에 닿지 못했습니다)")
