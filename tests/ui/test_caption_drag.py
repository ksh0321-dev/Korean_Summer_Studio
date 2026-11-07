"""자막 끌기: 타임라인 자막 블록 (옮기기·길이 바꾸기, 되돌리기 한 번) · 미리보기 자막 (위아래 자리)."""

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ks_studio.core.project import Project
from ks_studio.core.subtitles import CaptionStyle, Cue, Subtitles, Word


class FakeSources(QObject):
    """타임라인이 쓰는 원본 캐시 자리 (클립이 없어 거의 안 쓴다)."""

    ready = Signal(object)

    def info(self, _path):
        return None


def timeline_with_cue():
    from ks_studio.ui.editor.timeline import TimelineWidget

    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(FakeSources())
    widget.resize(1200, 300)
    cue = Cue(2.0, 4.0, "안녕", "", (Word(2.0, 3.0, " 안"), Word(3.0, 4.0, " 녕")))
    project = Project(subtitles=Subtitles((cue,)))
    widget._project = project
    widget.snapping = False
    widget.show()
    app.processEvents()
    return app, widget, project


def drag(widget, start: QPointF, end: QPointF) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start.toPoint())
    for step in range(1, 6):
        pos = start + (end - start) * step / 5
        QTest.mouseMove(widget, pos.toPoint())
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end.toPoint())


def test_timeline_caption_move_and_resize():
    app, widget, project = timeline_with_cue()
    edits = []
    widget.edited.connect(edits.append)
    rect = widget._caption_rect(project, project.subtitles.cues[0])
    one_second = widget._ppf * project.fps
    drag(widget, rect.center(), rect.center() + QPointF(one_second, 0))  # 몸통을 1초 오른쪽으로
    assert len(edits) == 1  # 놓을 때 한 번만 (되돌리기 한 번)
    moved = edits[0].subtitles.cues[0]
    assert abs(moved.start - 3.0) < 0.1 and abs(moved.end - 5.0) < 0.1 and abs(moved.words[0].start - 3.0) < 0.1
    widget._project = project
    drag(widget, QPointF(rect.right() - 2, rect.center().y()), QPointF(rect.right() - 2 + one_second, rect.center().y()))
    longer = edits[1].subtitles.cues[0]
    assert longer.start == 2.0 and abs(longer.end - 5.0) < 0.1  # 오른쪽 끝을 끌면 길이만


def test_preview_caption_drag_changes_bottom():
    from ks_studio.ui.widgets.video_view import VideoView

    app = QApplication.instance() or QApplication([])
    view = VideoView(mirrored=False)
    view.resize(360, 640)
    view.set_caption_draggable(True)
    view.show_rgb(np.zeros((1920, 1080, 3), np.uint8))
    view.set_caption("안녕하세요", CaptionStyle(bottom=0.1))
    view.show()
    view.grab()  # 한 번 그려야 자막 자리를 안다
    app.processEvents()
    moved = []
    view.caption_moved.connect(moved.append)
    rect = view._caption_rect
    assert rect is not None and view.caption_hit(rect.center())
    drag(view, rect.center(), rect.center() - QPointF(0, 640 * 0.2))  # 화면 높이의 20% 위로
    assert len(moved) == 1 and abs(moved[0] - 0.3) < 0.03
    assert not view.caption_hit(QPointF(5, 5))
