"""무음 구간 찾기 (말하는 영상에서 숨 고르기·침묵을 잘라내는 용도). 순수 계산 — Qt·파일 접근 없음.

1) silent_spans: 원본 소리의 10ms 피크에서 '기준보다 조용한 시간이 충분히 긴 곳'을 찾는다 (원본 오디오 시각, 초)
2) timeline_ranges: 그 구간을 V1 클립 위치로 옮겨 타임라인 프레임 구간으로 바꾼다 (오디오 오프셋 반영)
잘라내기는 core.edits.delete_ranges (구간 지우기를 뒤에서부터 반복)가 한다.
"""

import math
from collections.abc import Sequence

from ks_studio.core.project import MAIN_TRACK, Project

PEAK_SECONDS = 0.01  # services.sources.compute_peaks의 창 길이


def silent_spans(peaks: Sequence[float], threshold_db: float, min_ms: int, pad_ms: int) -> list[tuple[float, float]]:
    """peaks(10ms마다 최대 진폭 0~1)가 threshold_db 아래로 min_ms 이상 이어진 구간. 앞뒤로 pad_ms씩은 남긴다 (말끝이 잘리지 않게)."""
    limit = 10 ** (threshold_db / 20)
    pad = pad_ms / 1000
    spans, start = [], None
    for i, value in enumerate([*peaks, math.inf]):  # 끝에 '시끄러운 값'을 하나 붙여 마지막 구간도 닫는다
        if value < limit:
            if start is None:
                start = i
            continue
        if start is not None:
            if (i - start) * PEAK_SECONDS * 1000 >= min_ms:
                lo, hi = start * PEAK_SECONDS + pad, i * PEAK_SECONDS - pad
                if hi > lo:
                    spans.append((lo, hi))
            start = None
    return spans


def timeline_ranges(project: Project, spans_by_clip: dict[str, list[tuple[float, float]]]) -> list[tuple[int, int]]:
    """V1 클립별 무음(원본 오디오 시각) → 타임라인 프레임 구간 [시작, 끝). 겹치거나 붙은 구간은 합친다."""
    ranges = []
    for placed in project.placed():
        clip = placed.clip
        if placed.track != MAIN_TRACK or clip.id not in spans_by_clip:
            continue
        # 클립 안 위치 L에서 들리는 소리 = 원본 오디오 시각 source_time(L) - 오프셋 → 거꾸로 풀면 L
        base = clip.in_frame / clip.fps - clip.audio_offset_ms / 1000
        for lo, hi in spans_by_clip[clip.id]:
            a = max(0, round((lo - base) * project.fps))
            b = min(placed.end - placed.start, round((hi - base) * project.fps))
            if a < b:
                ranges.append((placed.start + a, placed.start + b))
    merged: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged
