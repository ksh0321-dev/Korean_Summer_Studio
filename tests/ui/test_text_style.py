"""글자 모양: 자막과 같은 칸(글꼴·테두리 두께·배경 상자 진하기), 프리셋 저장·불러오기(자막과 같은 목록), 모든 글자에 입히기."""

from dataclasses import replace

from PySide6.QtWidgets import QApplication, QInputDialog

from ks_studio.core.project import TextItem
from ks_studio.core.subtitles import CaptionStyle, load_style


def panel_with(tmp_path, *texts):
    from ks_studio.ui.editor.text_panel import TextPanel

    app = QApplication.instance() or QApplication([])
    panel = TextPanel()
    panel.styles_dir = tmp_path
    panel._fill_styles()
    panel.set_texts(tuple(texts))
    edits = []
    panel.edited.connect(edits.append)
    return app, panel, edits


def test_new_style_controls_change_selected_text(tmp_path):
    _app, panel, edits = panel_with(tmp_path, TextItem(0.0, 2.0, "a", id="a"), TextItem(1.0, 3.0, "b", id="b"))
    panel.select("b")
    assert panel.font_combo.isEnabled() and panel.apply_all.isEnabled()
    panel.outline_spin.setValue(0)
    panel.box_check.setChecked(True)
    panel.box_spin.setValue(30)
    font = panel.font_combo.itemData(1)
    panel.font_combo.setCurrentIndex(1)
    b = next(t for t in edits[-1] if t.id == "b")
    a = next(t for t in edits[-1] if t.id == "a")
    assert b.style.outline_width == 0 and b.style.box and abs(b.style.box_opacity - 0.3) < 1e-6 and b.style.font == font
    assert a.style == TextItem(0, 1, "").style  # 고르지 않은 글자는 그대로


def test_apply_to_all_keeps_place(tmp_path):
    big = replace(TextItem(0, 1, "").style, size=0.2, color="#ff0000", box=True)
    _app, panel, edits = panel_with(tmp_path, TextItem(0.0, 2.0, "a", x=0.1, y=0.2, style=big, id="a"),
                                    TextItem(1.0, 3.0, "b", x=0.8, y=0.9, id="b"))
    panel.select("a")
    panel._apply_to_all()
    assert len(edits) == 1
    b = next(t for t in edits[0] if t.id == "b")
    assert b.style == big and (b.x, b.y) == (0.8, 0.9)


def test_preset_save_load_shared_with_captions(tmp_path, monkeypatch):
    style = replace(TextItem(0, 1, "").style, color="#00ff00", outline_width=0.05)
    _app, panel, edits = panel_with(tmp_path, TextItem(0.0, 2.0, "a", style=style, id="a"), TextItem(1.0, 3.0, "b", id="b"))
    panel.select("a")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("제목", True)))
    panel._save_style()
    saved = load_style(tmp_path, "제목")
    assert saved.color == "#00ff00" and saved.bottom == CaptionStyle().bottom  # 자막에서 불러와도 맨 아래에 붙지 않게
    panel.set_texts(edits[-1] if edits else panel._texts)
    panel.select("b")
    panel._load_style(panel.style_combo.findData("제목"))
    b = next(t for t in edits[-1] if t.id == "b")
    assert b.style.color == "#00ff00" and b.style.outline_width == 0.05 and b.style.bottom == 0.0


def test_template_multiline_and_fade_controls(tmp_path, monkeypatch):
    from ks_studio.core.subtitles import STYLE_TEMPLATES
    from ks_studio.ui.editor.text_panel import TEMPLATE, from_cell, to_cell

    assert from_cell(to_cell("첫 줄\n둘째 줄")) == "첫 줄\n둘째 줄"
    _app, panel, edits = panel_with(tmp_path, TextItem(0.0, 2.0, "a", x=0.3, id="a"))
    panel.select("a")
    name = next(iter(STYLE_TEMPLATES))
    panel._load_style(panel.style_combo.findData(TEMPLATE + name))
    assert edits[-1][0].style == STYLE_TEMPLATES[name] and edits[-1][0].x == 0.3
    assert not panel.style_delete.isEnabled()  # 기본 제공 모양은 지울 수 없다
    panel.fade_in_spin.setValue(400)
    panel.shadow_spin.setValue(50)
    panel.align_combo.setCurrentIndex(panel.align_combo.findData("left"))
    item = edits[-1][0]
    assert item.fade_in_ms == 400 and item.style.shadow == 0.5 and item.style.align == "left"
    monkeypatch.setattr(QInputDialog, "getMultiLineText", staticmethod(lambda *a, **k: ("위\n아래\n", True)))
    panel._edit_multiline()
    assert edits[-1][0].text == "위\n아래"
    assert panel.table.item(0, 2).text() == "위 ⏎ 아래"


def test_caption_panel_has_same_look_controls():
    from ks_studio.ui.editor.caption_panel import CaptionPanel

    QApplication.instance() or QApplication([])
    panel = CaptionPanel()
    got = []
    panel.edited.connect(got.append)
    panel.shadow_spin.setValue(70)
    panel.italic_check.setChecked(True)
    assert got and got[-1].style.shadow == 0.7 and got[-1].style.italic
