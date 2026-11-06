"""메인 창: 상단 바(프로젝트 메뉴 + 단계 버튼 + 장치 상태 + 설정) + 4단계 페이지 스택 + 하단 상태 표시줄.

프로젝트 = 폴더. 녹화와 컷편집 모두 지금 연 프로젝트 폴더를 쓴다.
"""

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ks_studio import __version__
from ks_studio.app.paths import app_relative, from_app_relative
from ks_studio.core.project import Project
from ks_studio.core.settings import AppSettings, DeviceRef, with_recent_project
from ks_studio.core.workspace import Workspace
from ks_studio.services.cleanup import clean_project
from ks_studio.services.devices import DeviceMonitor, find_camera, find_microphone
from ks_studio.services.sources import SourceCache
from ks_studio.ui.beautify.beautify_page import BeautifyPage
from ks_studio.ui.editor.editor_page import EditorPage
from ks_studio.ui.export.export_page import ExportPage
from ks_studio.ui.recorder.close_wait_dialog import CloseWaitDialog
from ks_studio.ui.recorder.recorder_page import RecorderPage
from ks_studio.ui.settings.settings_dialog import SettingsDialog
from ks_studio.ui.widgets.status_bar import INFO, WARNING, LogBridge, StatusBar

log = logging.getLogger(__name__)


EDITOR_STAGE = 1


def _device_problem(label: str, ref: DeviceRef | None, find: Callable) -> str | None:
    if ref is None:
        return f"{label} 미설정"
    if find(ref) is None:
        return f"{label} 연결 안 됨 ({ref.name})"
    return None


class MainWindow(QMainWindow):
    def __init__(
        self, settings: AppSettings, settings_path: Path, default_workspace: Path, devices: DeviceMonitor
    ) -> None:
        super().__init__()
        self.resize(1600, 960)
        self.settings = settings
        self._settings_path = settings_path
        self._default_workspace = default_workspace
        self._devices = devices
        self._sources = SourceCache(self)
        self.recorder_page = RecorderPage(devices)
        self.editor_page = EditorPage(self._sources)
        self.beautify_page = BeautifyPage(self._sources, self.editor_page.optimizer)
        self.export_page = ExportPage(self._sources)

        self.stack = QStackedWidget()
        self.stage_buttons = QButtonGroup(self)
        self.stage_buttons.setExclusive(True)

        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        bar = QHBoxLayout(top_bar)
        bar.setContentsMargins(12, 6, 12, 6)

        title = QLabel("KS Studio")
        title.setObjectName("appTitle")
        bar.addWidget(title)
        bar.addSpacing(12)
        self.project_button = QPushButton()
        self.project_button.setMenu(self._build_project_menu())
        bar.addWidget(self.project_button)
        bar.addSpacing(24)

        pages = [
            ("녹화", self.recorder_page),
            ("컷편집", self.editor_page),
            ("AI 보정", self.beautify_page),
            ("내보내기", self.export_page),
        ]
        for index, (name, page) in enumerate(pages):
            self.stack.addWidget(page)
            button = QPushButton(name)
            button.setObjectName("stageButton")
            button.setCheckable(True)
            self.stage_buttons.addButton(button, index)
            bar.addWidget(button)
        self.stage_buttons.idClicked.connect(self._show_stage)
        self.stage_buttons.button(0).setChecked(True)

        bar.addStretch(1)
        self.device_status = QLabel()
        self.device_status.setObjectName("deviceStatus")
        bar.addWidget(self.device_status)
        bar.addSpacing(12)
        self.settings_button = QPushButton("⚙ 설정")
        self.settings_button.clicked.connect(self._open_settings)
        bar.addWidget(self.settings_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        for page in (self.recorder_page, self.editor_page, self.beautify_page, self.export_page):
            page.message.connect(self.status_bar.show_message)
            page.indicator.connect(self.status_bar.set_indicator)
        self._log_bridge = LogBridge(self)  # 경고·오류 로그도 상태 표시줄로
        self._log_bridge.message.connect(self.status_bar.show_message)
        logging.getLogger("ks_studio").addHandler(self._log_bridge.handler)

        devices.cameras_changed.connect(self._update_device_status)
        devices.microphones_changed.connect(self._update_device_status)
        self.recorder_page.busy_changed.connect(self._set_busy)
        self.recorder_page.settings_edited.connect(self._apply_settings)
        self.recorder_page.edit_requested.connect(self._edit_recording)
        self._update_device_status()
        self._remember_project()
        self._update_title()
        self._clean_project()
        self.recorder_page.activate(self.settings, self._workspace())
        QTimer.singleShot(0, self.recorder_page.offer_recovery)

    def closeEvent(self, event) -> None:
        """편집을 저장하고, 녹화 중이면 멈추고, 마무리할 녹화가 남았으면 끝날 때까지 기다린 뒤 닫는다 (취소 불가)."""
        self.editor_page.deactivate()
        self.beautify_page.deactivate()
        self.export_page.deactivate()
        if self.recorder_page.begin_close():
            event.ignore()
            dialog = CloseWaitDialog(self.recorder_page, self)
            dialog.exec()
            dialog.deleteLater()
            QTimer.singleShot(0, self.close)
            return
        logging.getLogger("ks_studio").removeHandler(self._log_bridge.handler)
        super().closeEvent(event)

    def _workspace(self) -> Workspace:
        # 설정에는 앱 폴더 기준 상대 경로로 적는다 (앱 폴더를 통째로 옮겨도 프로젝트가 따라온다)
        return Workspace(from_app_relative(self.settings.workspace_dir) if self.settings.workspace_dir else self._default_workspace)

    # ── 단계 전환 ────────────────────────────────────────────────

    def _show_stage(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.stage_buttons.button(index).setChecked(True)
        current = self.stack.currentWidget()
        if current is not self.recorder_page:
            self.recorder_page.deactivate()
        if current is not self.editor_page:
            self.editor_page.deactivate()
        if current is not self.beautify_page:
            self.beautify_page.deactivate()
        if current is not self.export_page:
            self.export_page.deactivate()
        if current is self.recorder_page:
            self.recorder_page.activate(self.settings, self._workspace())
        elif current is self.editor_page:
            self.editor_page.activate(self.settings, self._workspace())
        elif current is self.beautify_page:
            self.beautify_page.activate(self._workspace())
        elif current is self.export_page:
            self.export_page.activate(self._workspace())

    def _edit_recording(self, path: Path) -> None:
        self._show_stage(EDITOR_STAGE)
        self.editor_page.append_media(path)

    def _set_busy(self, busy: bool) -> None:
        """녹화(카운트다운 포함) 중에는 카메라가 끊기지 않게 단계 전환·프로젝트·설정을 잠근다."""
        for button in self.stage_buttons.buttons():
            button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.project_button.setEnabled(not busy)

    # ── 프로젝트 ────────────────────────────────────────────────

    def _build_project_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction("새 프로젝트…", self._new_project)
        menu.addAction("프로젝트 열기…", self._choose_project)
        recent = menu.addMenu("최근 프로젝트")
        recent.aboutToShow.connect(lambda: self._fill_recent_menu(recent))
        return menu

    def _fill_recent_menu(self, menu: QMenu) -> None:
        menu.clear()
        for folder in self.settings.recent_projects:
            path = from_app_relative(folder)
            action = menu.addAction(f"{path.name}    —    {path}", lambda p=path: self._open_project(p))
            action.setEnabled(path.is_dir())
        if menu.isEmpty():
            menu.addAction("(없음)").setEnabled(False)

    def _new_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "새 프로젝트 폴더 선택 (새 폴더를 만들어 고르세요)")
        if not folder:
            return
        workspace = Workspace(Path(folder))
        if not workspace.project_file.exists():
            Project().save(workspace.project_file)
        self._open_project(workspace.root)

    def _choose_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 열기", str(self._workspace().root))
        if folder:
            self._open_project(Path(folder))

    def _open_project(self, folder: Path) -> None:
        self.editor_page.save()
        recent = with_recent_project(self.settings.recent_projects, app_relative(folder))
        self._apply_settings(replace(self.settings, workspace_dir=app_relative(folder), recent_projects=recent))
        self._update_title()
        self._clean_project()
        self._show_stage(self.stack.currentIndex())  # 지금 보이는 페이지를 새 프로젝트로 다시 연다

    def _clean_project(self) -> None:
        """프로젝트를 열 때 쓰지 않는 사본·캐시·쓰다 만 파일을 지운다 (영상 파생 파일은 커서 그냥 두면 쌓인다)."""
        try:
            freed = clean_project(self._workspace())
        except OSError:
            log.exception("프로젝트 정리 실패")
            return
        if freed > 1_000_000:
            self.status_bar.show_message(f"쓰지 않는 사본·캐시 {freed / 1e6:.0f}MB를 정리했습니다", INFO)

    def _remember_project(self) -> None:
        folder = app_relative(self._workspace().root)
        if self.settings.recent_projects[:1] != [folder]:
            self._apply_settings(
                replace(self.settings, recent_projects=with_recent_project(self.settings.recent_projects, folder))
            )

    def _update_title(self) -> None:
        name = self._workspace().root.name
        self.setWindowTitle(f"KS Studio {__version__} — {name}")
        self.project_button.setText(f"프로젝트: {name}")

    # ── 설정 ────────────────────────────────────────────────────

    def _apply_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings.save(self._settings_path)
        self._update_device_status()

    def _open_settings(self) -> None:
        self.recorder_page.deactivate()  # 카메라는 한 곳에서만 열 수 있다
        dialog = SettingsDialog(self.settings, self._devices, self, shortcuts=self.editor_page.shortcut_catalog())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_settings(dialog.selected_settings())
        dialog.deleteLater()
        self._show_stage(self.stack.currentIndex())

    def _update_device_status(self) -> None:
        problems = [
            p
            for p in (
                _device_problem("카메라", self.settings.camera, find_camera),
                _device_problem("마이크", self.settings.microphone, find_microphone),
            )
            if p
        ]
        if problems:
            text = "⚠ " + " · ".join(problems) + " — 설정에서 확인"
        else:
            text = f"카메라: {self.settings.camera.name}  ·  마이크: {self.settings.microphone.name}"
        self.device_status.setText(text)
        self.device_status.setProperty("warning", bool(problems))
        self.device_status.style().polish(self.device_status)
        calibration = self.settings.sync_calibration
        if problems:
            self.status_bar.set_indicator("sync", "")
        elif calibration is None:  # 처음 쓰는 카메라·마이크 조합
            self.status_bar.set_indicator("sync", "⚠ 싱크 보정 필요 — 녹화 화면에서 '박수로 싱크 보정'", WARNING)
        else:
            self.status_bar.set_indicator("sync", f"싱크 보정 {calibration.audio_delay_ms:+d}ms", INFO)
