import math

from ks_studio.core.cfr import FrameSlotter

FPS = 30
TICK = 0.015625  # Windows 타이머 틱 (실측 지연 단위)


def place_all(arrivals, settle_frames=10):
    slotter = FrameSlotter(FPS, settle_frames)
    return [slotter.place(a) for a in arrivals], slotter


def tick_quantized(n, camera_fps=FPS):
    """카메라는 일정 간격으로 찍지만 도착은 다음 15.6ms 틱으로 밀린다 (실측 31/47ms 간격 재현)."""
    return [math.ceil((i / camera_fps) / TICK) * TICK for i in range(n)]


def duplicates(placed):
    return sum(len(slots) - 1 for slots in placed if slots)


def test_steady_frames_fill_one_slot_each():
    placed, slotter = place_all([i / FPS for i in range(50)])
    assert placed == [[i] for i in range(50)]
    assert slotter.dropped == 0


def test_tick_quantized_arrivals_have_no_duplicates_or_drops():
    placed, slotter = place_all(tick_quantized(600))
    assert duplicates(placed) == 0
    assert slotter.dropped == 0


def test_60fps_camera_with_timer_ticks_has_no_duplicates_or_drops():
    """60fps에선 한 칸(16.7ms)이 Windows 틱(15.6ms)과 거의 같다 → 틱 흔들림만으로는 복제·버림이 없어야 한다."""
    slotter = FrameSlotter(60, settle_frames=120)
    stamps = [math.ceil((i / 60) / TICK) * TICK for i in range(1200)]
    placed = [slotter.place(s) for s in stamps]
    assert duplicates(placed) == 0
    assert slotter.dropped == 0


def test_single_spike_late_is_jitter():
    arrivals = [i / FPS for i in range(40)]
    arrivals[25] += 0.030
    placed, slotter = place_all(arrivals)
    assert duplicates(placed) == 0
    assert slotter.dropped == 0


def test_slow_camera_gets_duplicates_only():
    n = 650  # 실측: 29.946fps로 약 21.7초
    placed, slotter = place_all(tick_quantized(n, camera_fps=29.946))
    total_slots = sum(len(s) for s in placed)
    expected_slots = round((n - 1) / 29.946 * FPS) + 1
    assert slotter.dropped == 0
    assert abs(total_slots - expected_slots) <= 1


def test_fast_camera_gets_drops_only():
    placed, slotter = place_all([i / (FPS * 1.01) for i in range(300)])
    assert duplicates(placed) == 0
    assert slotter.dropped >= 2


def test_missing_frame_is_filled_by_duplicate():
    arrivals = [i / FPS for i in range(60) if i != 30]
    placed, slotter = place_all(arrivals)
    assert duplicates(placed) == 1
    assert slotter.dropped == 0
    assert slotter.next_slot == 60


def test_camera_fps_drop_keeps_frames_at_their_time():
    """실측 버그: 어두워져 카메라가 15fps로 떨어지면 영상이 최대 0.3초 앞당겨졌다."""
    stamps = []
    t = 0.0
    for interval, count in ((1 / 30, 90), (1 / 15, 30), (1 / 30, 60)):  # 3초 정상 → 2초 15fps → 2초 정상
        for _ in range(count):
            stamps.append(math.ceil(t / TICK) * TICK)
            t += interval
    slotter = FrameSlotter(FPS, settle_frames=10)
    worst = 0.0
    for stamp in stamps:
        slots = slotter.place(stamp)
        if slots:
            worst = max(worst, abs(slots[-1] - (stamp - slotter.origin) * FPS))
    assert worst < 1.0  # 어떤 프레임도 제 시각에서 한 칸(33ms) 넘게 벗어나지 않는다


def test_settle_pulls_origin_back_to_earliest_phase():
    _, slotter = place_all([0.015, 1 / FPS, 2 / FPS])
    assert slotter.origin == 0.0
