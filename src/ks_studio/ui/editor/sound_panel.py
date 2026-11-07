"""컷편집 오른쪽 '소리' 탭: 소리 줄(배경 음악 · 떼어 낸 클립 소리) 목록과 고른 조각의 볼륨·페이드·덕킹.

조각은 타임라인의 소리 띠에서 끌어 옮기고 양 끝을 끌어 자른다. 배경 음악은 말소리가 나는 동안 저절로 작아진다(덕킹).
"""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.project import Sound

AUDIO_FILTER = "소리 (*.mp3 *.m4a *.aac *.wav *.flac *.ogg *.mp4 *.mov)"


class SoundPanel(QWidget):
    edited = Signal(object)  # 새 소리 줄 (tuple[Sound])
    add_requested = Signal()  # '음악 넣기…' (파일 고르기·넣기는 편집 화면이)
    seek_requested = Signal(int)  # 타임라인 프레임

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sounds: tuple[Sound, ...] = ()
        self._fps = 30
        add = QPushButton("♪ 음악 넣기…")
        add.setToolTip("배경 음악 파일을 재생 헤드 자리에 넣습니다 (mp3·m4a·wav 등)")
        add.clicked.connect(self.add_requested.emit)
        delete = QPushButton("삭제")
        delete.clicked.connect(self._delete)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(delete)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._show)
        self.list.itemClicked.connect(lambda item: self.seek_requested.emit(self._sounds[self.list.row(item)].start))
        self.volume = QDoubleSpinBox()
        self.volume.setRange(-40.0, 12.0)
        self.volume.setDecimals(1)
        self.volume.setSuffix(" dB")
        self.fade_in, self.fade_out = QSpinBox(), QSpinBox()
        for box in (self.fade_in, self.fade_out):
            box.setRange(0, 10000)
            box.setSingleStep(250)
            box.setSuffix(" ms")
        self.duck = QCheckBox("말할 때 저절로 작게 (배경 음악)")
        self.duck.setToolTip("클립에서 말소리가 나는 동안 이 소리를 -12dB로 줄였다가, 말이 끝나면 서서히 되돌립니다")
        self.volume.valueChanged.connect(lambda v: self._change(volume_db=v))
        self.fade_in.valueChanged.connect(lambda v: self._change(fade_in_ms=v))
        self.fade_out.valueChanged.connect(lambda v: self._change(fade_out_ms=v))
        self.duck.toggled.connect(lambda on: self._change(duck=on))
        form = QFormLayout()
        form.addRow("볼륨", self.volume)
        form.addRow("페이드 인", self.fade_in)
        form.addRow("페이드 아웃", self.fade_out)
        form.addRow("", self.duck)
        hint = QLabel("타임라인 위 '소리' 띠에서 조각을 끌어 옮기고, 양 끝을 끌어 자릅니다. 클립 소리는 '속성'의 '소리 떼어 내기'로 여기에 옵니다.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.list, 1)
        layout.addLayout(form)
        layout.addWidget(hint)
        self._show(-1)

    def set_sounds(self, sounds: tuple[Sound, ...], fps: int) -> None:
        if sounds == self._sounds and fps == self._fps:
            return
        current = self.list.currentRow()
        current_id = self._sounds[current].id if 0 <= current < len(self._sounds) else None
        self._sounds, self._fps = sounds, fps
        self.list.blockSignals(True)
        self.list.clear()
        for s in sounds:
            kind = "음악" if s.duck else "떼어 낸 소리"
            self.list.addItem(f"{Path(s.source).name}  ·  {s.start / fps:.1f}초부터 {s.length_ms / 1000:.1f}초  ({kind})")
        self.list.blockSignals(False)
        self.select(current_id)

    def select(self, sound_id: str | None) -> None:
        row = next((i for i, s in enumerate(self._sounds) if s.id == sound_id), -1)
        self.list.setCurrentRow(row)
        self._show(row)

    def _show(self, row: int) -> None:
        sound = self._sounds[row] if 0 <= row < len(self._sounds) else None
        for w, value in ((self.volume, sound.volume_db if sound else 0.0), (self.fade_in, sound.fade_in_ms if sound else 0),
                         (self.fade_out, sound.fade_out_ms if sound else 0)):
            w.setEnabled(sound is not None)
            w.blockSignals(True)
            w.setValue(value)
            w.blockSignals(False)
        self.duck.setEnabled(sound is not None)
        self.duck.blockSignals(True)
        self.duck.setChecked(bool(sound and sound.duck))
        self.duck.blockSignals(False)

    def _change(self, **changes) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._sounds):
            sounds = (*self._sounds[:row], replace(self._sounds[row], **changes), *self._sounds[row + 1 :])
            self._sounds = sounds
            self.edited.emit(sounds)

    def _delete(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._sounds):
            self.edited.emit(self._sounds[:row] + self._sounds[row + 1 :])
