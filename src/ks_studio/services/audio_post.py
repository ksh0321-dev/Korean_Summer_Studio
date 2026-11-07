"""내보낼 때 소리 다듬기: 잡음 줄이기(FFmpeg afftdn) → 소리 크기 맞추기(loudnorm, 유튜브 기준 -14 LUFS).

타임라인 소리를 프레임마다 밀어 넣고 나오는 만큼 받아 인코더로 넘긴다. loudnorm은 앞뒤를 보며 맞추느라 조금 늦게 나오고
안에서 192kHz로 계산하므로 끝에 48kHz로 되돌린다.
"""

from fractions import Fraction

import av
import av.filter
import numpy as np

from ks_studio.services.sources import AUDIO_RATE

LOUDNESS = -14.0  # 유튜브·인스타그램이 맞추는 크기 (LUFS) — 그보다 작으면 작게, 크면 줄여서 튼다
TRUE_PEAK = -1.5
NOISE_REDUCTION_DB = 12  # 잡음 줄이기 세기 (afftdn nr). 너무 크면 목소리가 물속처럼 뭉개진다


class AudioPost:
    def __init__(self, denoise: bool, loudness: bool, rate: int = AUDIO_RATE) -> None:
        self._rate = rate
        graph = av.filter.Graph()
        nodes = [graph.add_abuffer(format="flt", sample_rate=rate, layout="mono", time_base=Fraction(1, rate))]
        if denoise:
            nodes.append(graph.add("afftdn", f"nr={NOISE_REDUCTION_DB}:nf=-40"))
        if loudness:
            nodes.append(graph.add("loudnorm", f"I={LOUDNESS}:TP={TRUE_PEAK}:LRA=11"))
            nodes.append(graph.add("aresample", str(rate)))
        nodes.append(graph.add("aformat", f"sample_fmts=flt:channel_layouts=mono:sample_rates={rate}"))
        nodes.append(graph.add("abuffersink"))
        graph.link_nodes(*nodes).configure()
        self._graph = graph
        self._pushed = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """소리 한 덩어리(float32 모노)를 넣고, 다 된 만큼 받는다 (처음엔 비어 있을 수 있다)."""
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples.reshape(1, -1), np.float32), format="flt", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = self._rate, self._pushed, Fraction(1, self._rate)
        self._pushed += len(samples)
        self._graph.push(frame)
        return self._drain()

    def flush(self) -> list[np.ndarray]:
        """남은 소리 모두."""
        self._graph.push(None)
        return self._drain()

    def _drain(self) -> list[np.ndarray]:
        out = []
        while True:
            try:
                out.append(self._graph.pull().to_ndarray().reshape(-1).astype(np.float32))
            except (av.BlockingIOError, av.EOFError):
                return out


def loudness_of(samples: np.ndarray, rate: int = AUDIO_RATE) -> float:
    """통합 소리 크기 (LUFS, FFmpeg ebur128) — 확인용."""
    graph = av.filter.Graph()
    nodes = [graph.add_abuffer(format="flt", sample_rate=rate, layout="mono", time_base=Fraction(1, rate)),
             graph.add("ebur128", "metadata=1"), graph.add("abuffersink")]
    graph.link_nodes(*nodes).configure()
    value = None
    for i in range(0, len(samples), rate // 10):
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[None, i : i + rate // 10], np.float32), format="flt",
                                           layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = rate, i, Fraction(1, rate)
        graph.push(frame)
        while True:
            try:
                out = graph.pull()
            except (av.BlockingIOError, av.EOFError):
                break
            found = out.metadata.get("lavfi.r128.I")
            if found is not None:
                value = float(found)
    return value if value is not None else float("-inf")
