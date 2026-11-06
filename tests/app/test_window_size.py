"""처음 창 크기: 모니터 비율 그대로, 작업 영역 안."""

from ks_studio.app.application import initial_window_size


def test_window_follows_monitor_aspect():
    w, h = initial_window_size((1920, 1080), (1920, 1040))  # 16:9 + 작업 표시줄
    assert abs(w / h - 16 / 9) < 0.01 and w <= 1920 and h <= 1040
    w, h = initial_window_size((3413, 1440), (3413, 1400))  # 21:9 울트라와이드
    assert abs(w / h - 3413 / 1440) < 0.01 and h <= 1400 * 0.85 + 1
    w, h = initial_window_size((1440, 2560), (1440, 2520))  # 세로 모니터
    assert w < h and w <= 1440
