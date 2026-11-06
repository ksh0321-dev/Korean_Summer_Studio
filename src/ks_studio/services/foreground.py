"""재생 중에는 백그라운드 작업을 잠깐 멈춘다 — 재생이 CPU·디스크를 먼저 쓰게.

썸네일(영상 전체 디코딩)·편집용 사본·AI 분석/보정본이 재생과 같은 CPU를 나눠 쓰면 미리보기가 끊겼다.
플레이어(컷편집·자막 탭·AI 1분 미리보기·내보내기 미리보기)는 재생을 시작/멈출 때 set_playing을 부르고,
오래 도는 작업은 프레임마다 yield_to_playback()을 부른다 (재생 중이 아니면 바로 지나간다).

재생 자신이 기다리는 일(재생할 소리 풀기)은 멈추면 안 된다 — 거기엔 부르지 않는다.
"""

import threading

_lock = threading.Lock()
_players: set[str] = set()
_idle = threading.Event()
_idle.set()


def set_playing(player: str, playing: bool) -> None:
    with _lock:
        (_players.add if playing else _players.discard)(player)
        if _players:
            _idle.clear()
        else:
            _idle.set()


def is_playing() -> bool:
    return not _idle.is_set()


def yield_to_playback() -> None:
    """재생 중이면 멈출 때까지 기다린다."""
    _idle.wait()
