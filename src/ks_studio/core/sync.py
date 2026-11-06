"""박수로 A/V 싱크 재기. 순수 파이썬 — Qt import 금지."""

import statistics

INCONSISTENT_MS = 40  # 박수끼리 이보다 차이가 크면 다시 재는 게 낫다 (30fps 한 프레임 ≈ 33ms)


def contact_time(previous_frame_time: float, contact_frame_time: float) -> float:
    """손이 처음 닿아 보이는 프레임과 그 직전(다른 장면) 프레임 사이 어딘가에서 닿았다 → 가운데로 추정."""
    return (previous_frame_time + contact_frame_time) / 2


def estimate_delay_ms(offsets_s: list[float]) -> tuple[int, int]:
    """박수마다 (손이 닿은 시각 - 소리 시각) → (보정값 = 중앙값 ms, 박수끼리 차이 = 최대-최소 ms).

    보정값이 +면 영상이 소리보다 늦다 → 오디오를 그만큼 늦춰야 한다.
    """
    if not offsets_s:
        raise ValueError("박수가 없습니다")
    ms = [s * 1000 for s in offsets_s]
    return round(statistics.median(ms)), round(max(ms) - min(ms))
