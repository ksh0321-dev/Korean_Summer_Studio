"""모델 버튼: 없으면 '⬇ 이름 크기', 누르면 받고 '✓ 이름'. 실패하면 이유를 알리고 다시 누를 수 있다."""

import time

from PySide6.QtWidgets import QApplication

from ks_studio.ui.widgets.model_button import ModelButton


def wait(app, condition, seconds=5.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_download_then_ready():
    app = QApplication.instance() or QApplication([])
    have = []

    def download(on_progress):
        on_progress(0.5)
        have.append(True)

    button = ModelButton("몸매", 70, lambda: bool(have), download, "몸매 보정")
    assert button.text() == "⬇ 몸매 70MB" and button.isEnabled()
    results = []
    button.finished.connect(results.append)
    button.click()
    assert wait(app, lambda: results)
    assert results == [""] and button.text() == "✓ 몸매" and not button.isEnabled()


def test_failure_lets_user_retry_and_big_sizes_in_gb():
    app = QApplication.instance() or QApplication([])

    def download(on_progress):
        raise OSError("연결 끊김")

    button = ModelButton("자막", 3100, lambda: False, download, "자막 만들기")
    assert button.text() == "⬇ 자막 3.1GB"
    results = []
    button.finished.connect(results.append)
    button.click()
    assert wait(app, lambda: results)
    assert results == ["연결 끊김"] and button.isEnabled() and button.text() == "⬇ 자막 3.1GB"


def test_missing_package_disables():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    button = ModelButton("얼굴", 290, lambda: False, lambda p: None, "AI 보정")
    button.set_missing("AI 패키지가 없습니다")
    assert not button.isEnabled() and "패키지 없음" in button.text()
