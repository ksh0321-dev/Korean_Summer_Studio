"""타임라인 글자 띠: 블록 옮기기·길이 바꾸기 (놓을 때 한 번), 누르면 고르기·그 글자로 가기, 겹치면 칸 나누기."""

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from ks_studio.core.project import Project, TextItem
from tests.ui.test_caption_drag import FakeSources, drag


def timeline_with_texts(*texts):
    from ks_studio.ui.editor.timeline import TimelineWidget

    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(FakeSources())
    widget.resize(1200, 300)
    project = Project(texts=tuple(texts))
    widget._project = project
    widget.snapping = False
    widget.show()
    app.processEvents()
    return widget, project


def test_text_block_move_resize_and_click():
    widget, project = timeline_with_texts(TextItem(2.0, 4.0, "제목", id="t1"))
    edits, clicked, seeks = [], [], []
    widget.edited.connect(edits.append)
    widget.text_clicked.connect(clicked.append)
    widget.seek_requested.connect(seeks.append)
    rect = widget._text_rects(project)[0]
    one_second = widget._ppf * project.fps
    drag(widget, rect.center(), rect.center() + QPointF(one_second, 0))
    assert len(edits) == 1 and clicked == ["t1"]
    moved = edits[0].texts[0]
    assert abs(moved.start - 3.0) < 0.1 and abs(moved.end - 5.0) < 0.1
    widget._project = project
    drag(widget, QPointF(rect.left() + 2, rect.center().y()), QPointF(rect.left() + 2 - one_second, rect.center().y()))
    earlier = edits[1].texts[0]
    assert abs(earlier.start - 1.0) < 0.1 and earlier.end == 4.0  # 왼쪽 끝은 시작만
    widget._project = project
    drag(widget, rect.center(), rect.center())  # 끌지 않고 떼면 그 글자 시작으로
    assert len(edits) == 2 and seeks[-1] == round(2.0 * project.fps)


def test_overlapping_texts_get_lanes_and_cannot_go_negative():
    widget, project = timeline_with_texts(TextItem(0.5, 3.0, "a", id="a"), TextItem(1.0, 2.0, "b", id="b"),
                                          TextItem(3.0, 4.0, "c", id="c"))
    lanes, count = widget._text_lanes(project.texts)
    assert lanes == [0, 1, 0] and count == 2
    rects = widget._text_rects(project)
    assert rects[1].top() > rects[0].top()
    assert widget._text_at(rects[1].center().x(), rects[1].center().y()) == 1
    edits = []
    widget.edited.connect(edits.append)
    one_second = widget._ppf * project.fps
    drag(widget, rects[0].center(), rects[0].center() - QPointF(2 * one_second, 0))  # 0초 앞으로는 못 간다
    moved = next(t for t in edits[0].texts if t.id == "a")
    assert moved.start == 0.0 and abs(moved.end - 2.5) < 0.1
