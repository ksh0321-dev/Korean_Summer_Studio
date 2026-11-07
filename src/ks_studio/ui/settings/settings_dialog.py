"""설정 다이얼로그: 카메라·마이크 선택, 실시간 미리보기, 레벨 미터, 미디어 가져오기 방식.

저장된 장치가 빠져 있으면 '연결 안 됨'으로 표시하고, 다른 장치로 몰래 바꾸지 않는다.
프로젝트(작업 폴더)는 상단 "프로젝트" 메뉴에서 고른다.
"""

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QTabWidget,
)

from ks_studio.core.settings import MEDIA_COPY, MEDIA_REFERENCE, AppSettings, DeviceRef, match_device
from ks_studio.services.capture import CameraPreview, MicInput
from ks_studio.services.devices import (
    DeviceMonitor,
    best_format,
    camera_refs,
    find_camera,
    find_microphone,
    microphone_refs,
)
from ks_studio.core.workspace import Workspace
from ks_studio.ui.settings.general_tabs import DefaultsTab, PerformanceTab, StorageTab
from ks_studio.ui.settings.shortcuts import ShortcutTab
from ks_studio.ui.widgets.level_meter import LevelMeter
from ks_studio.ui.widgets.video_view import VideoView


def _fill(combo: QComboBox, refs: list[DeviceRef], selected: DeviceRef | None) -> None:
    """목록을 채우고 selected를 고른다. selected가 없으면 맨 위에 '연결 안 됨' 항목으로 남긴다."""
    combo.blockSignals(True)
    combo.clear()
    for ref in refs:
        combo.addItem(ref.name, ref)
    index = match_device(refs, selected) if selected is not None else None
    if selected is not None and index is None:
        combo.insertItem(0, f"⚠ 연결 안 됨 — {selected.name}", selected)
        combo.setItemData(0, QColor("#ff6b6b"), Qt.ItemDataRole.ForegroundRole)
        index = 0
    combo.setCurrentIndex(-1 if index is None else index)
    combo.blockSignals(False)


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, devices: DeviceMonitor, parent: QWidget | None = None,
                 shortcuts: list | None = None, workspace: Workspace | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.resize(960, 680)
        self._settings = settings

        self.camera_combo = QComboBox()
        self.camera_combo.setPlaceholderText("카메라를 선택하세요")
        self.mic_combo = QComboBox()
        self.mic_combo.setPlaceholderText("마이크를 선택하세요")
        self.camera_info = QLabel()
        self.video_view = VideoView()
        self.level_meter = LevelMeter()

        self._preview = CameraPreview(self.video_view.sink, self)
        self._preview.error.connect(
            lambda message: self.video_view.set_message(
                f"카메라를 열 수 없음: {message}\n다른 앱이 카메라를 쓰고 있는지 확인"
            )
        )
        self._mic = MicInput(self)
        self._mic.level.connect(self.level_meter.set_peak)

        self.import_combo = QComboBox()
        self.import_combo.addItem("프로젝트 폴더로 복사 (폴더만 옮기면 다 따라감, 디스크를 더 씀)", MEDIA_COPY)
        self.import_combo.addItem("원본 경로만 참조 (복사 안 함, 원본을 옮기면 끊김)", MEDIA_REFERENCE)
        self.import_combo.setCurrentIndex(max(0, self.import_combo.findData(settings.media_import)))

        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        devices.cameras_changed.connect(self._refill_cameras)
        devices.microphones_changed.connect(self._refill_microphones)

        form = QFormLayout()
        form.addRow("카메라 (영상 입력)", self.camera_combo)
        form.addRow("마이크 (오디오 입력)", self.mic_combo)
        form.addRow("미디어 가져오기", self.import_combo)

        camera_column = QVBoxLayout()
        camera_column.addWidget(self.video_view, 1)
        camera_column.addWidget(self.camera_info)
        mic_column = QVBoxLayout()
        mic_column.addWidget(self.level_meter, 1)
        mic_column.addWidget(QLabel("마이크 레벨"), 0, Qt.AlignmentFlag.AlignHCenter)
        previews = QHBoxLayout()
        previews.addLayout(camera_column, 1)
        previews.addLayout(mic_column)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        devices_tab = QWidget()
        devices_layout = QVBoxLayout(devices_tab)
        devices_layout.addLayout(form)
        devices_layout.addLayout(previews, 1)
        tabs = QTabWidget()
        tabs.addTab(devices_tab, "장치 · 가져오기")
        self.shortcut_tab = ShortcutTab(shortcuts, settings.shortcuts) if shortcuts else None
        if self.shortcut_tab is not None:
            tabs.addTab(self.shortcut_tab, "단축키 (컷편집)")
        self.performance_tab = PerformanceTab(settings)
        self.defaults_tab = DefaultsTab(settings)
        tabs.addTab(self.performance_tab, "성능")
        tabs.addTab(StorageTab(workspace), "저장 공간")
        tabs.addTab(self.defaults_tab, "기본값")
        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)

        _fill(self.camera_combo, camera_refs(), settings.camera)
        _fill(self.mic_combo, microphone_refs(), settings.microphone)
        self._on_camera_changed()
        self._on_mic_changed()

    def selected_settings(self) -> AppSettings:
        return replace(
            self._settings,
            camera=self.camera_combo.currentData(),
            microphone=self.mic_combo.currentData(),
            media_import=self.import_combo.currentData(),
            shortcuts=self.shortcut_tab.overrides() if self.shortcut_tab is not None else self._settings.shortcuts,
            **self.performance_tab.apply(self._settings),
            **self.defaults_tab.apply(self._settings),
        )

    def accept(self) -> None:
        if self.shortcut_tab is not None and self.shortcut_tab.problems():
            QMessageBox.warning(self, "단축키", "같은 키를 쓰는 기능이 있습니다. 단축키 탭에서 하나를 바꿔 주세요.")
            return
        super().accept()

    def done(self, result: int) -> None:
        self._preview.stop()
        self._mic.stop()
        super().done(result)

    def _refill_cameras(self) -> None:
        _fill(self.camera_combo, camera_refs(), self.camera_combo.currentData())
        self._on_camera_changed()

    def _refill_microphones(self) -> None:
        _fill(self.mic_combo, microphone_refs(), self.mic_combo.currentData())
        self._on_mic_changed()

    def _on_camera_changed(self) -> None:
        ref = self.camera_combo.currentData()
        device = find_camera(ref)
        if device is None:
            self._preview.stop()
            self.video_view.set_message("카메라를 선택하세요" if ref is None else "카메라가 연결되어 있지 않습니다")
            self.camera_info.setText("")
            return
        self.video_view.set_message("카메라 여는 중…")
        fmt = best_format(device)
        self._preview.start(device, fmt)
        if fmt is None:
            self.camera_info.setText("⚠ 이 카메라는 1920×1080을 지원하지 않음")
        else:
            self.camera_info.setText(f"미리보기: 1920×1080 @ {fmt.maxFrameRate():.0f}fps (이 카메라의 1080p 최대)")

    def _on_mic_changed(self) -> None:
        device = find_microphone(self.mic_combo.currentData())
        if device is None:
            self._mic.stop()
        else:
            self._mic.start(device)
