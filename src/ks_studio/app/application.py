"""조립 지점: QApplication을 만들고 테마를 입힌 뒤 메인 창을 띄운다."""

import logging
import logging.handlers
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QStyle

from ks_studio.app.fonts import family, register_fonts
from ks_studio import __version__
from ks_studio.app.paths import DEFAULT_WORKSPACE, FROZEN, SETTINGS_PATH, USER_DIR, migrate_user_files
from ks_studio.core.settings import AppSettings
from ks_studio.services.devices import DeviceMonitor
from ks_studio.ui.main_window import MainWindow

THEME_PATH = Path(__file__).resolve().parent.parent / "ui" / "theme" / "dark.qss"
WINDOW_FILL = 0.85  # 처음 창 크기 = 모니터 작업 영역의 85%


def initial_window_size(screen: tuple[int, int], available: tuple[int, int]) -> tuple[int, int]:
    """처음 창 크기: 모니터와 같은 비율(16:9 모니터면 16:9, 21:9면 21:9)로 작업 영역(작업 표시줄 제외) 안에 들어가게."""
    aspect = screen[0] / screen[1]
    width = available[0] * WINDOW_FILL
    height = width / aspect
    if height > available[1] * WINDOW_FILL:
        height = available[1] * WINDOW_FILL
        width = height * aspect
    return round(width), round(height)


def _setup_logging() -> None:
    """소스로 돌리면 터미널에, exe면(터미널이 없다) user/logs/ks_studio.log에 남긴다 — 다른 PC에서 문제를 볼 때 이 파일을 본다."""
    handlers = None
    if FROZEN:
        log_dir = USER_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers = [logging.handlers.RotatingFileHandler(log_dir / "ks_studio.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger(__name__).info("KS Studio %s 시작 (%s)", __version__, "exe" if FROZEN else "소스")


def main() -> int:
    _setup_logging()
    if "--selftest" in sys.argv:  # 배포한 PC에서 GPU·모델·인코더가 도는지 확인 (user/logs/selftest.txt)
        from ks_studio.app.selftest import run

        return run()
    app = QApplication(sys.argv)
    app.setApplicationName("KS Studio")
    register_fonts()  # 테마(qss)가 Pretendard를 쓰므로 먼저
    app.setFont(QFont(family(), 10))  # qss가 닿지 않는 직접 그리는 글씨(타임라인 등)도 같은 글꼴로
    app.setStyleSheet(THEME_PATH.read_text(encoding="utf-8"))

    migrate_user_files()  # 예전 위치의 개인 설정 → user/
    window = MainWindow(AppSettings.load(SETTINGS_PATH), SETTINGS_PATH, DEFAULT_WORKSPACE, DeviceMonitor(app))
    # 위치를 Windows에 맡기면 모니터가 없는 좌표(실측: x=-1714)에 뜰 수 있어 주 모니터 가운데에 고정한다
    screen = app.primaryScreen()
    available = screen.availableGeometry()
    window.resize(*initial_window_size((screen.geometry().width(), screen.geometry().height()), (available.width(), available.height())))
    window.setGeometry(QStyle.alignedRect(Qt.LayoutDirection.LeftToRight, Qt.AlignmentFlag.AlignCenter, window.size(), available))
    window.show()
    return app.exec()
