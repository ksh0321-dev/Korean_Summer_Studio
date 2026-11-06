"""재생 중엔 백그라운드 작업이 기다리고, 모든 플레이어가 멈추면 다시 간다."""

import threading
import time

from ks_studio.services.foreground import is_playing, set_playing, yield_to_playback


def test_background_waits_while_any_player_plays():
    set_playing("editor", True)
    set_playing("subtitle", True)
    done = threading.Event()
    threading.Thread(target=lambda: (yield_to_playback(), done.set()), daemon=True).start()
    time.sleep(0.1)
    assert is_playing() and not done.is_set()
    set_playing("editor", False)
    time.sleep(0.05)
    assert not done.is_set()  # 자막 탭이 아직 재생 중
    set_playing("subtitle", False)
    assert done.wait(1) and not is_playing()
    yield_to_playback()  # 재생 중이 아니면 바로 지나간다
