"""하이라이트 분할 설정 창: 민감도 · 앞뒤 여유 · 최소 길이.

값을 움직이면 바로 options_changed로 알린다 → 편집 화면이 점수 줄(기준 점선·초록 띠)에 바로 보여 준다.
이미 나눈 컷은 그대로 — 새 기준으로 다시 나누려면 Ctrl+Z로 분할을 되돌리고 버튼을 다시 누른다.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.highlight_cuts import DEFAULT_MIN_LENGTH, DEFAULT_PAD, DEFAULT_SENSITIVITY, Options


class HighlightDialog(QDialog):
    options_changed = Signal(object)  # Options

    def __init__(self, options: Options, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("하이라이트 분할 설정")
        self.setMinimumWidth(420)
        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setValue(round(options.sensitivity))
        self.sensitivity.setToolTip("높이면 조금만 두드러져도 하이라이트로, 낮추면 아주 뚜렷한 곳만")
        self.sensitivity_value = QLabel()
        self.sensitivity_value.setMinimumWidth(28)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("적게"))
        slider_row.addWidget(self.sensitivity, 1)
        slider_row.addWidget(QLabel("많이"))
        slider_row.addWidget(self.sensitivity_value)
        self.pad = self._seconds(0.0, 3.0, options.pad, 0.5)
        self.pad.setToolTip("하이라이트 앞뒤로 이만큼 바깥에서 자릅니다 (말이 끊긴 곳에 ±0.5초 맞춤)")
        self.min_length = self._seconds(1.0, 15.0, options.min_length, 1.0)
        self.min_length.setToolTip("이보다 짧게 튀는 곳은 하이라이트로 치지 않습니다")
        form = QFormLayout()
        form.addRow("민감도", slider_row)
        form.addRow("앞뒤 여유", self.pad)
        form.addRow("최소 길이", self.min_length)
        hint = QLabel("바꾼 기준은 타임라인 맨 아래 점수 줄(점선·초록 띠)에 바로 보입니다.\n"
                      "이미 나눈 컷에 다시 적용하려면 Ctrl+Z로 분할을 되돌린 뒤 '✨ 하이라이트 분할'을 다시 누르세요.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("확인")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("기본값")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.sensitivity.valueChanged.connect(self._changed)
        self.pad.valueChanged.connect(self._changed)
        self.min_length.valueChanged.connect(self._changed)
        self.sensitivity_value.setText(str(self.sensitivity.value()))

    def _seconds(self, low: float, high: float, value: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(1)
        box.setSingleStep(step)
        box.setSuffix(" 초")
        box.setValue(value)
        return box

    def options(self) -> Options:
        return Options(float(self.sensitivity.value()), self.pad.value(), self.min_length.value())

    def _changed(self, _value=None) -> None:
        self.sensitivity_value.setText(str(self.sensitivity.value()))
        self.options_changed.emit(self.options())

    def _defaults(self) -> None:
        self.sensitivity.setValue(DEFAULT_SENSITIVITY)
        self.pad.setValue(DEFAULT_PAD)
        self.min_length.setValue(DEFAULT_MIN_LENGTH)


LOW_DEFAULT = 30  # 점수 낮은 컷 지우기의 처음 기준


class LowScoreDialog(QDialog):
    """점수 낮은 컷 지우기: 기준 점수를 고르면 지워질 컷 수·길이를 바로 보여 준다. 고른 기준은 threshold()."""

    def __init__(self, means: list[float], fps: int, frames_below, parent: QWidget | None = None) -> None:
        """means = 컷 평균 점수들, frames_below(기준) = 그 기준으로 지워질 (컷 수, 프레임 수)."""
        super().__init__(parent)
        self.setWindowTitle("점수 낮은 컷 지우기")
        self.setMinimumWidth(400)
        self._fps, self._frames_below = fps, frames_below
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(LOW_DEFAULT)
        self.value = QLabel()
        self.value.setMinimumWidth(48)
        row = QHBoxLayout()
        row.addWidget(self.slider, 1)
        row.addWidget(self.value)
        self.summary = QLabel()
        scores = " · ".join(f"{m:.0f}" for m in means[:40]) + (" …" if len(means) > 40 else "")
        hint = QLabel(f"컷 평균 점수가 기준보다 낮은 V1 컷을 지우고 뒤를 당깁니다 (Ctrl+Z로 되돌리기).\n지금 컷 점수: {scores}")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("지우기")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        form = QFormLayout()
        form.addRow("기준 점수", row)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.summary)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.slider.valueChanged.connect(self._update)
        self._update()

    def threshold(self) -> float:
        return float(self.slider.value())

    def _update(self, _value=None) -> None:
        self.value.setText(f"{self.slider.value()}점")
        count, frames = self._frames_below(self.threshold())
        self.summary.setText(f"{self.slider.value()}점보다 낮은 컷 {count}개 · {frames / self._fps:.1f}초를 지웁니다" if count
                             else f"{self.slider.value()}점보다 낮은 컷이 없습니다")
        self._ok.setEnabled(count > 0)
