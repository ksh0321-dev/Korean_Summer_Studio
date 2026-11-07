"""단축키 이름·겹침 찾기, 설정 탭이 바꾼 것만 돌려주기."""

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from ks_studio.ui.settings.shortcuts import ShortcutTab, conflicts, shortcut_name


def test_names_and_conflicts():
    assert shortcut_name("✂ 분할") == "분할" and shortcut_name("뒤 자르기 ⇥") == "뒤 자르기"
    assert shortcut_name("재생/정지") == "재생/정지" and shortcut_name("[ 구간 시작") == "구간 시작"
    assert conflicts({"a": "Ctrl+B", "b": "ctrl+b", "c": "Q", "d": ""}) == {"Ctrl+B": ["a", "b"]}


def test_tab_returns_only_changed_keys():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    catalog = [("분할", "자르기", ("Ctrl+B",)), ("다시 하기", "다시", ("Ctrl+Shift+Z", "Ctrl+Y")), ("재생/정지", "재생", ("Space",))]
    tab = ShortcutTab(catalog, {"재생/정지": ["K"]})
    assert tab.overrides() == {"재생/정지": ["K"]}  # 저장된 것은 그대로, 나머지는 기본
    tab.edits["분할"].setKeySequence(QKeySequence("K"))
    assert tab.problems() == {"K": ["분할", "재생/정지"]}
    tab.edits["분할"].clear()
    assert tab.overrides() == {"분할": [], "재생/정지": ["K"]} and not tab.problems()
    tab._reset()
    assert tab.overrides() == {}


def test_editor_shortcut_names_are_unique_and_not_blank(tmp_path):
    """컷편집 단축키 목록: 이름이 비거나 겹치면 설정 창에 빈 줄이 생기고 한쪽이 사라진다 (－·＋ 확대/축소가 그랬다)."""
    from PySide6.QtWidgets import QApplication

    from ks_studio.services.sources import SourceCache
    from ks_studio.ui.editor.editor_page import EditorPage

    app = QApplication.instance() or QApplication([])  # noqa: F841
    sources = SourceCache()
    page = EditorPage(sources)
    names = [name for name, _tip, _keys in page.shortcut_catalog()]
    assert all(names) and len(names) == len(set(names))
    assert {"축소", "확대"} <= set(names)
