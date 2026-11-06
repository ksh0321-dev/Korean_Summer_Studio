"""속성 패널: 선택한 클립의 정보, 오디오 오프셋(싱크 맞추기), 볼륨·페이드, 화면(크기·위치·투명도), 디졸브.

볼륨·페이드·화면·디졸브는 선택한 클립 전부에 적용된다 (편집 화면이 처리). 크기·위치·투명도는 %로 보여 준다.
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGridLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from ks_studio.core.project import Clip
from ks_studio.ui.editor.timeline import format_frames

OFFSET_LIMIT_MS = 2000
VOLUME_RANGE_DB = (-40.0, 12.0)
FADE_LIMIT_MS = 10000


class Inspector(QWidget):
    offset_changed = Signal(str, int)  # 클립 id, ms
    prop_changed = Signal(str, float)  # 바꾼 Clip 속성 이름 (volume_db·fade_in_ms·scale·pos_x·…), 값 (scale 등은 0~1 비율)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clip: Clip | None = None
        self._fps = 30
        self.source_label = QLabel("-")
        self.source_label.setWordWrap(True)
        self.range_label = QLabel("-")
        self.offset = QSpinBox()
        self.offset.setRange(-OFFSET_LIMIT_MS, OFFSET_LIMIT_MS)
        self.offset.setSuffix(" ms")
        self.offset.valueChanged.connect(self._emit_offset)
        self.volume = QDoubleSpinBox()
        self.volume.setRange(*VOLUME_RANGE_DB)
        self.volume.setDecimals(1)
        self.volume.setSingleStep(1.0)
        self.volume.setSuffix(" dB")
        self.volume.valueChanged.connect(lambda value: self.prop_changed.emit("volume_db", value))
        self.fade_in, self.fade_out = QSpinBox(), QSpinBox()
        for name, box in (("fade_in_ms", self.fade_in), ("fade_out_ms", self.fade_out)):
            box.setRange(0, FADE_LIMIT_MS)
            box.setSingleStep(100)
            box.setSuffix(" ms")
            box.valueChanged.connect(lambda value, n=name: self.prop_changed.emit(n, float(value)))
        # 화면 속 화면 (%)
        self._screen: dict[str, QSpinBox] = {}
        for name, low, high in (("scale", 5, 400), ("pos_x", -50, 150), ("pos_y", -50, 150), ("opacity", 0, 100)):
            box = QSpinBox()
            box.setRange(low, high)
            box.setSuffix(" %")
            box.valueChanged.connect(lambda value, n=name: self.prop_changed.emit(n, value / 100))
            self._screen[name] = box
        self.transition = QSpinBox()
        self.transition.setRange(0, FADE_LIMIT_MS)
        self.transition.setSingleStep(100)
        self.transition.setSuffix(" ms")
        self.transition.setToolTip("V1에서 앞 클립과 겹치며 서서히 바뀌는 시간 (0 = 바로 바뀜). 겹친 만큼 전체가 짧아집니다")
        self.transition.valueChanged.connect(lambda value: self.prop_changed.emit("transition_ms", float(value)))

        nudges = QGridLayout()
        self._nudge_buttons = []
        for column, (label, delta) in enumerate((("-1프레임", None), ("-10ms", -10), ("+10ms", 10), ("+1프레임", None))):
            button = QPushButton(label)
            sign = -1 if label.startswith("-") else 1
            button.clicked.connect(lambda _=False, d=delta, s=sign: self._nudge(d if d is not None else s * round(1000 / self._fps)))
            nudges.addWidget(button, 0, column)
            self._nudge_buttons.append(button)

        form = QFormLayout()
        form.addRow("원본", self.source_label)
        form.addRow("구간", self.range_label)
        form.addRow("오디오 오프셋", self.offset)
        form.addRow("볼륨", self.volume)
        form.addRow("페이드 인", self.fade_in)
        form.addRow("페이드 아웃", self.fade_out)
        form.addRow("크기", self._screen["scale"])
        form.addRow("위치 가로", self._screen["pos_x"])
        form.addRow("위치 세로", self._screen["pos_y"])
        form.addRow("투명도", self._screen["opacity"])
        form.addRow("앞 디졸브", self.transition)
        hint = QLabel("+면 소리를 늦춥니다. 박수 소리와 손이 닿는 장면이 맞도록 조정하세요.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("속성"))
        layout.addLayout(form)
        layout.addLayout(nudges)
        layout.addWidget(hint)
        layout.addStretch(1)
        self.show_clip(None, 30)

    def show_clip(self, clip: Clip | None, fps: int) -> None:
        self._clip, self._fps = clip, fps
        enabled = clip is not None
        boxes = (self.offset, self.volume, self.fade_in, self.fade_out, self.transition, *self._screen.values())
        for widget in (*boxes, *self._nudge_buttons):
            widget.setEnabled(enabled)
        values = [
            (self.offset, clip.audio_offset_ms if clip else 0),
            (self.volume, clip.volume_db if clip else 0.0),
            (self.fade_in, clip.fade_in_ms if clip else 0),
            (self.fade_out, clip.fade_out_ms if clip else 0),
            (self.transition, clip.transition_ms if clip else 0),
        ]
        values += [(box, round(getattr(clip, name) * 100) if clip else 100) for name, box in self._screen.items()]
        for box, value in values:
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        if clip is None:
            self.source_label.setText("클립을 선택하세요")
            self.range_label.setText("-")
            return
        self.source_label.setText(f"{Path(clip.source).name}  ({clip.fps:g}fps)")
        # 원본 구간을 타임라인 fps 기준 시각으로 보여준다 (원본 fps가 달라도 같은 눈금)
        start, end = (round(f / clip.fps * fps) for f in (clip.in_frame, clip.out_frame))
        self.range_label.setText(
            f"{format_frames(start, fps)} ~ {format_frames(end, fps)}  ({format_frames(end - start, fps)})"
        )

    def _nudge(self, delta_ms: int) -> None:
        self.offset.setValue(self.offset.value() + delta_ms)

    def _emit_offset(self, value: int) -> None:
        if self._clip is not None:
            self.offset_changed.emit(self._clip.id, value)
