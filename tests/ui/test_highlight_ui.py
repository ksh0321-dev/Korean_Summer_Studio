"""하이라이트 분할 화면: 버튼·도움말 · 점수 줄(비었을 때·그린 뒤) · 설정 창 · 저장되는 설정."""

import numpy as np
from PySide6.QtWidgets import QApplication, QToolButton

from ks_studio.core.highlight import AUDIO_STEP, Signals
from ks_studio.core.highlight_cuts import HighlightBook, Options
from ks_studio.core.project import Clip, Project
from ks_studio.core.settings import AppSettings
from ks_studio.services.sources import SourceCache
from ks_studio.ui.editor.highlight_dialog import HighlightDialog
from ks_studio.ui.editor.timeline import SCORE_H, TimelineWidget


def app():
    return QApplication.instance() or QApplication([])


def signals(duration: float = 20.0, loud=(6.0, 15.0)) -> Signals:
    db = np.full(round(duration / AUDIO_STEP), -40.0, np.float32)
    db[round(loud[0] / AUDIO_STEP) : round(loud[1] / AUDIO_STEP)] = -10
    return Signals(duration, db)


def test_button_sits_next_to_silence_cut_with_menu_and_v1_help():
    app()
    from ks_studio.ui.editor.editor_page import EditorPage

    page = EditorPage(SourceCache())
    buttons = {b.defaultAction().text(): b for b in page.findChildren(QToolButton) if b.defaultAction()}
    button = buttons["✨ 하이라이트 분할"]
    assert button.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert button.y() == buttons["🔇 무음 자르기…"].y()  # 둘째 줄, 무음 자르기 옆
    assert "V1(맨 아래 영상 줄)" in page.highlight_action.toolTip()
    page.apply_shortcuts({})  # 단축키를 다시 입혀도 긴 도움말은 그대로
    assert "V1(맨 아래 영상 줄)" in page.highlight_action.toolTip()
    names = [name for name, _tip, _keys in page.shortcut_catalog()]
    assert "하이라이트 분할" in names


def test_no_v1_clip_says_so():
    app()
    from ks_studio.ui.editor.editor_page import EditorPage

    page = EditorPage(SourceCache())
    said = []
    page.message.connect(lambda text, _level: said.append(text))
    page.highlight_action.trigger()
    assert said and "V1" in said[-1] and not page._highlight_worker.running


def test_score_band_is_blank_until_scores_then_tells_score(tmp_path):
    app()
    from ks_studio.core.workspace import Workspace

    timeline = TimelineWidget(SourceCache())
    timeline.resize(1000, 400)
    project = Project(clips=(Clip("a.mp4", 0, 600, id="A"),))
    timeline.set_project(project, Workspace(tmp_path))
    timeline.fit()
    x, y = timeline._x(300), timeline.height() - SCORE_H / 2
    assert timeline._score_tip(x, y) is None  # 분할 전: 빈 줄
    assert timeline._track_at(y) is None  # 점수 줄은 영상 줄이 아니다 (누르면 재생 헤드 이동)
    image = timeline.grab().toImage()
    assert image.width() > 0
    timeline.set_highlights(HighlightBook({"a.mp4": signals()}))
    tip = timeline._score_tip(x, y)
    assert tip is not None and "점" in tip and "기준" in tip
    timeline.grab()  # 그리기에서 죽지 않는다


def test_dialog_reports_options_and_resets():
    app()
    dialog = HighlightDialog(Options(sensitivity=20, pad=2.0, min_length=5.0))
    seen = []
    dialog.options_changed.connect(seen.append)
    assert dialog.options() == Options(20, 2.0, 5.0)
    dialog.sensitivity.setValue(80)
    assert seen[-1].sensitivity == 80
    dialog._defaults()
    assert dialog.options() == Options()


def test_settings_keep_highlight_options(tmp_path):
    path = tmp_path / "settings.json"
    AppSettings(highlight_sensitivity=70, highlight_pad=1.5, highlight_min_length=4.0).save(path)
    loaded = AppSettings.load(path)
    assert (loaded.highlight_sensitivity, loaded.highlight_pad, loaded.highlight_min_length) == (70, 1.5, 4.0)


def test_menu_actions_keep_drop_select():
    app()
    from ks_studio.core.highlight_cuts import split_project
    from ks_studio.core.history import History
    from ks_studio.ui.editor.editor_page import EditorPage

    page = EditorPage(SourceCache())
    said = []
    page.message.connect(lambda text, _level: said.append(text))
    assert not page.keep_highlights_action.isEnabled()  # 분할 전에는 못 누른다
    book = HighlightBook({"a.mp4": signals()})
    project = split_project(Project(clips=(Clip("a.mp4", 0, 600, id="A"),)), book).project
    page._history = History(project)
    page._on_highlight_done({})  # 점수가 생긴 상태 (이미 나뉘어 있어 더 자르지 않는다)
    page._highlights = book
    page.timeline.set_highlights(book)
    assert page.keep_highlights_action.isEnabled() and page.select_highlights_action.isEnabled()
    middle = page.project.clips[1].id
    page.select_highlights_action.trigger()
    assert page.timeline.selection == {middle}
    page.keep_highlights_action.trigger()
    assert [c.id for c in page.project.clips] == [middle] and "하이라이트만 남기기" in said[-1]
    page.undo()
    assert len(page.project.clips) == 3
    page._clear_highlights()
    assert not page.drop_low_action.isEnabled()


def test_low_score_dialog_counts():
    app()
    from ks_studio.ui.editor.highlight_dialog import LowScoreDialog

    dialog = LowScoreDialog([10, 50, 90], 30, lambda t: (sum(m < t for m in (10, 50, 90)), 300), None)
    dialog.slider.setValue(60)
    assert dialog.threshold() == 60 and "2개" in dialog.summary.text() and dialog._ok.isEnabled()
    dialog.slider.setValue(5)
    assert not dialog._ok.isEnabled()
