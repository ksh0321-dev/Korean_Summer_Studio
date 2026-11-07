"""설정 다이얼로그의 '성능' · '저장 공간' · '기본값' 탭."""

import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.app.paths import CAPTION_STYLES_DIR, MODELS_DIR
from ks_studio.core.presets import list_presets
from ks_studio.core.settings import PREVIEW_QUALITIES, AppSettings
from ks_studio.core.workspace import Workspace
from ks_studio.services.accel import open_folder
from ks_studio.services.deliver import PRESETS


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("hint")
    return label


def folder_size(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def human_size(size: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f}{unit}"
    return f"{size}B"


class PerformanceTab(QWidget):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview_combo = QComboBox()
        for key, (label, _size) in PREVIEW_QUALITIES.items():
            self.preview_combo.addItem(label, key)
        self.preview_combo.setCurrentIndex(max(0, self.preview_combo.findData(settings.preview_quality)))
        self.software_check = QCheckBox("내보내기 · AI 보정을 CPU 인코더로만 (느림)")
        self.software_check.setChecked(settings.software_encoding)
        self.software_check.setToolTip("그래픽 카드 인코더(NVENC 등)에서 오류가 나거나 결과물이 깨질 때만 켜세요. 녹화는 그대로 그래픽 카드를 씁니다.")
        form = QFormLayout()
        form.addRow("재생 중 미리보기 화질", self.preview_combo)
        form.addRow("", self.software_check)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(_hint(
            "컷편집 재생이 끊기면 미리보기 화질을 낮추세요. 멈춘 장면과 내보낸 영상의 화질은 그대로입니다.\n"
            "AI 보정은 NVIDIA 그래픽 카드(CUDA)가 있으면 저절로 씁니다."
        ))
        layout.addStretch(1)

    def apply(self, settings: AppSettings) -> dict:
        return {"preview_quality": self.preview_combo.currentData(), "software_encoding": self.software_check.isChecked()}


class StorageTab(QWidget):
    """프로젝트 폴더 안 큰 것들과 AI 모델의 크기. 캐시는 비울 수 있다 (다시 만든다)."""

    def __init__(self, workspace: Workspace | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        rows: list[tuple[str, Path, bool]] = []
        if workspace is not None:
            rows += [
                ("녹화본", workspace.recordings_dir, False),
                ("가져온 미디어 (복사본)", workspace.media_dir, False),
                ("AI 보정본", workspace.ai_dir, False),
                ("내보낸 영상", workspace.exports_dir, False),
                ("자동 백업 (편집 기록)", workspace.root / "backups", False),
                ("캐시 (파형 · 썸네일 · 미리보기용 사본)", workspace.cache_dir, True),
            ]
        rows.append(("AI 모델 (모든 프로젝트 공통)", MODELS_DIR, False))
        grid = QGridLayout()
        self._sizes: dict[Path, QLabel] = {}
        for row, (name, folder, clearable) in enumerate(rows):
            grid.addWidget(QLabel(name), row, 0)
            size = QLabel()
            self._sizes[folder] = size
            grid.addWidget(size, row, 1)
            open_button = QPushButton("폴더 열기")
            open_button.clicked.connect(lambda _=False, f=folder: open_folder(f) if f.exists() else None)
            grid.addWidget(open_button, row, 2)
            if clearable:
                clear = QPushButton("비우기")
                clear.setToolTip("지워도 필요할 때 다시 만듭니다 (처음 열 때 잠깐 느려질 수 있음)")
                clear.clicked.connect(lambda _=False, f=folder: self._clear(f))
                grid.addWidget(clear, row, 3)
        grid.setColumnStretch(0, 1)
        box = QGroupBox(f"이 프로젝트: {workspace.root}" if workspace is not None else "저장 공간")
        box.setLayout(grid)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(_hint(
            "프로젝트를 열 때 쓰지 않는 사본·캐시는 저절로 정리됩니다. 녹화본·내보낸 영상은 직접 지우세요 (폴더 열기).\n"
            "자동 백업은 5분마다 최근 30개만 남깁니다 (프로젝트 메뉴 → 백업에서 되살리기)."
        ))
        layout.addStretch(1)
        self._measure()

    def _measure(self) -> None:
        for folder, label in self._sizes.items():
            label.setText(human_size(folder_size(folder)))

    def _clear(self, folder: Path) -> None:
        shutil.rmtree(folder, ignore_errors=True)  # 지금 열려 있는 파일은 남는다
        self._measure()


class DefaultsTab(QWidget):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.caption_combo = QComboBox()
        self.caption_combo.addItem("기본 모양", "")
        for name in list_presets(CAPTION_STYLES_DIR):
            self.caption_combo.addItem(name, name)
        self.caption_combo.setCurrentIndex(max(0, self.caption_combo.findData(settings.default_caption_style)))
        self.export_combo = QComboBox()
        self.export_combo.addItem("직접 고르기", "")
        for key, (label, _values) in PRESETS.items():
            self.export_combo.addItem(label, key)
        self.export_combo.setCurrentIndex(max(0, self.export_combo.findData(settings.default_export_preset)))
        self.update_check = QCheckBox("켤 때 새 버전이 있는지 확인 (GitHub)")
        self.update_check.setChecked(settings.check_updates)
        form = QFormLayout()
        form.addRow("새 프로젝트 자막 모양", self.caption_combo)
        form.addRow("내보내기 프리셋", self.export_combo)
        form.addRow("", self.update_check)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(_hint(
            "자막 모양 프리셋은 컷편집 '자막' 탭에서 저장합니다. 내보내기 프리셋은 내보내기 탭을 열 때 처음 한 번 골라 둡니다."
        ))
        layout.addStretch(1)

    def apply(self, settings: AppSettings) -> dict:
        return {
            "default_caption_style": self.caption_combo.currentData(),
            "default_export_preset": self.export_combo.currentData(),
            "check_updates": self.update_check.isChecked(),
        }
