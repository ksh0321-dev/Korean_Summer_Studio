"""효과 세기 칸: 슬라이더와 숫자 칸이 같이 움직이고, 바뀔 때 한 번만 알린다."""

from PySide6.QtWidgets import QApplication

from ks_studio.ui.beautify.effect_control import EffectControl


def test_slider_and_number_follow_each_other_once():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    control = EffectControl(-100)
    seen = []
    control.valueChanged.connect(seen.append)
    control.spin.setValue(-40)  # 숫자 칸에 입력
    assert control.slider.value() == -40 and control.value() == -40
    control.slider.setValue(25)  # 슬라이더를 끌기
    assert control.spin.value() == 25
    assert seen == [-40, 25]
    control.blockSignals(True)  # 영상을 바꿀 때 값을 보여 주기만
    control.setValue(70)
    control.blockSignals(False)
    assert control.spin.value() == 70 and seen == [-40, 25]
    control.spin.setValue(150)
    assert control.value() == 100  # 범위 밖은 끝값
