"""앱 글꼴: Pretendard (SIL Open Font License 1.1 — 무료, 프로그램에 넣어 배포 가능, 영상에 자막으로 써도 된다).

assets/fonts/에 Regular·SemiBold·Bold를 넣어 두고 앱을 켤 때 등록한다 (사용자 PC에 설치할 필요 없음).
화면 글꼴과 자막(내보내기에 굽는 글씨)이 같은 글꼴을 쓴다. 등록을 못 하면 맑은 고딕으로 그린다.
"""

from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FAMILY = "Pretendard"
FALLBACK = "Malgun Gothic"
_loaded: bool | None = None


def register_fonts() -> bool:
    """한 번만 등록한다 (QGuiApplication이 있어야 한다). 성공하면 True."""
    global _loaded
    if _loaded is None:
        from PySide6.QtGui import QFontDatabase

        ids = [QFontDatabase.addApplicationFont(str(path)) for path in sorted(FONT_DIR.glob("*.otf"))]
        _loaded = bool(ids) and all(i >= 0 for i in ids)
    return _loaded


def family() -> str:
    return FAMILY if register_fonts() else FALLBACK
