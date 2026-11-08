"""색 보정 탭: 영상 고르기 → 장면 → 값 바꾸기(저장) · 비교 그림 세 가지 · 기본 모양 · 모든 영상에 같게."""

import time

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from ks_studio.core.grade import BUILTIN_LOOKS, ColorGrade, build_lut, load_color_state
from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from tests.services.test_render_tracks import make_source


def pump(app, condition, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end and not condition():
        app.processEvents()
        time.sleep(0.01)


def test_compare_image_modes():
    from ks_studio.ui.color.color_page import GAP, compare_image

    rgb = np.full((90, 160, 3), 100, np.uint8)
    table = build_lut(ColorGrade(exposure=0.5))
    side = compare_image(rgb, table, "side", (400, 200))
    assert side.shape == (90, 160 * 2 + GAP, 3) and side[45, 10, 0] == 100 and side[45, -10, 0] > 120
    split = compare_image(rgb, table, "split", (400, 200))
    assert split.shape == (90, 160, 3) and split[45, 10, 0] == 100 and split[45, 150, 0] > 120
    after = compare_image(rgb, table, "after", (80, 45))
    assert after.shape == (45, 80, 3) and after[20, 40, 0] > 120


def test_page_edits_and_saves(tmp_path, monkeypatch):
    from ks_studio.services.optimize import Optimizer
    from ks_studio.services.sources import SourceCache
    from ks_studio.ui.color.color_page import BUILTIN, ColorPage

    app = QApplication.instance() or QApplication([])
    ws = Workspace(tmp_path)
    ws.recordings_dir.mkdir()
    make_source(ws.recordings_dir / "20261108_100000.mp4", 0, 30, 30)
    make_source(ws.recordings_dir / "20261108_110000.mp4", 1, 30, 30)
    Project().save(ws.project_file)
    page = ColorPage(SourceCache(), Optimizer())
    page.looks_dir = tmp_path / "looks"
    page.resize(1200, 700)
    page.show()
    page.activate(ws)
    assert page.list.count() == 2
    page.list.setCurrentRow(0)
    pump(app, lambda: page._frame is not None)
    assert page._frame is not None and page.center.currentIndex() == 1
    page.sliders["temperature"].setValue(40)
    page.hsl_group.button(2).click()
    page.sliders["hsl_sat"].setValue(-50)
    source = page._source
    grade = page._state.grades[source]
    assert grade.temperature == 0.4 and grade.hsl[2] == (0.0, -0.5, 0.0) and grade.hsl[0] == (0.0, 0.0, 0.0)
    page._load_look(page.look_combo.findData(BUILTIN + "흑백"))
    assert page._state.grades[source] == BUILTIN_LOOKS["흑백"] and page.sliders["saturation"].value() == -100
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    page._copy_to_all()
    page.deactivate()
    saved = load_color_state(ws.color_settings_file)
    assert len(saved.grades) == 2 and all(g == BUILTIN_LOOKS["흑백"] for g in saved.grades.values())
    page._reset()
    page.deactivate()
    assert len(load_color_state(ws.color_settings_file).grades) == 1  # 처음으로 = 저장에서 빠진다
