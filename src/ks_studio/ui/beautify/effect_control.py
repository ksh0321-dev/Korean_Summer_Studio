"""효과 하나의 세기: 슬라이더 + 숫자 칸 (어느 쪽을 바꿔도 같이 움직인다)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSlider, QSpinBox, QWidget


class EffectControl(QWidget):
    """값은 정수 %(-100~100 또는 0~100). 숫자 칸은 Enter·포커스를 옮길 때 적용 (치는 도중엔 그리지 않게)."""

    valueChanged = Signal(int)

    def __init__(self, minimum: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, 100)
        self.spin = QSpinBox()
        self.spin.setRange(minimum, 100)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(56)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        # 숫자 칸 → 슬라이더 → (숫자 칸 맞추고 알림). 같은 값이면 Qt가 신호를 다시 보내지 않아 한 번만 알린다
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._on_slider)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def _on_slider(self, value: int) -> None:
        self.spin.setValue(value)
        self.valueChanged.emit(value)
