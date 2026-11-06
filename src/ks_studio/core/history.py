"""되돌리기/다시 하기. Project가 불변이라 상태 스냅샷을 쌓기만 하면 된다 (클립 목록은 수 KB)."""

from ks_studio.core.project import Project

LIMIT = 200


class History:
    def __init__(self, project: Project) -> None:
        self.present = project
        self._undo: list[Project] = []
        self._redo: list[Project] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def push(self, project: Project) -> bool:
        """새 상태를 기록한다. 바뀐 게 없으면 False."""
        if project == self.present:
            return False
        self._undo = (self._undo + [self.present])[-LIMIT:]
        self._redo.clear()
        self.present = project
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.present)
        self.present = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.present)
        self.present = self._redo.pop()
        return True
