"""녹화 마무리(MKV → MP4)와 미디어 정보 조회. PyAV(FFmpeg 내장)만 쓴다."""

import logging
import os
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
AUDIO_BITRATE = 192_000
AUDIO_DELAY_TAG = "KS_AUDIO_DELAY_MS"  # 녹화 MKV 태그: 카메라·마이크 싱크 보정값 (+면 오디오를 늦춤)


def _audio_delay_ms(container) -> int:
    for key, value in container.metadata.items():
        if key.upper() == AUDIO_DELAY_TAG:
            return int(value)
    return 0


def _mono_frame(samples: np.ndarray) -> av.AudioFrame:
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1).astype(np.float32), format="fltp", layout="mono")
    frame.sample_rate = AUDIO_RATE
    return frame


def finalize_recording(
    partial: Path, output: Path, on_progress: Callable[[float, float | None], None] | None = None
) -> None:
    """H.264는 그대로 복사, PCM은 AAC 48kHz 모노로 변환. 성공하면 partial을 지운다.

    MKV 타임스탬프는 ms 단위라 33/34ms가 섞이므로, 프레임 번호로 다시 매겨 MP4를 정확한 CFR로 만든다.
    비정상 종료로 끝이 잘린 MKV는 읽을 수 있는 데까지 살린다.
    MKV에 싱크 보정값이 있으면 오디오를 그만큼 늦춘다(앞에 무음) / 앞당긴다(앞을 버림).
    on_progress(처리한 초, 전체 초 — 끝이 잘려 모르면 None)는 영상 1초 분량마다 불린다.
    """
    tmp = output.with_name(output.name + ".part")
    with av.open(str(partial)) as src, av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
        v_in, a_in = src.streams.video[0], src.streams.audio[0]
        frame_tb = 1 / Fraction(v_in.average_rate)
        total = src.duration / 1_000_000 if src.duration else None
        video_packets = 0
        v_out = dst.add_stream_from_template(v_in)
        v_out.time_base = frame_tb  # MP4 시간 단위를 1/fps의 배수로 → 프레임 간격이 정확히 같아진다
        a_out = dst.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        a_out.bit_rate = AUDIO_BITRATE
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=AUDIO_RATE)
        audio_pts = 0
        delay = round(_audio_delay_ms(src) * AUDIO_RATE / 1000)
        skip = max(0, -delay)  # 앞당기기: 원본 앞부분을 이만큼 버린다

        def encode_audio(frames) -> None:
            nonlocal audio_pts, skip
            for frame in frames:
                if skip:
                    samples = frame.to_ndarray()[0]
                    if len(samples) <= skip:
                        skip -= len(samples)
                        continue
                    frame, skip = _mono_frame(samples[skip:]), 0
                frame.pts, frame.time_base = audio_pts, Fraction(1, AUDIO_RATE)
                audio_pts += frame.samples
                dst.mux(a_out.encode(frame))

        if delay > 0:  # 늦추기: 앞에 무음을 넣는다
            encode_audio([_mono_frame(np.zeros(delay, np.float32))])

        try:
            for packet in src.demux(v_in, a_in):
                if packet.dts is None:
                    continue
                if packet.stream is v_in:
                    video_packets += 1
                    if on_progress and video_packets % round(1 / frame_tb) == 0:
                        on_progress(float(packet.pts * packet.time_base), total)
                    packet.pts = round(packet.pts * packet.time_base / frame_tb)
                    packet.dts = round(packet.dts * packet.time_base / frame_tb)
                    packet.duration = 1
                    packet.time_base = frame_tb
                    packet.stream = v_out
                    dst.mux(packet)
                else:
                    for frame in packet.decode():
                        encode_audio(resampler.resample(frame))
        except av.FFmpegError as e:
            log.warning("%s: 끝이 잘린 파일 — 읽은 데까지 저장 (%s)", partial.name, e)
        encode_audio(resampler.resample(None))
        dst.mux(a_out.encode(None))
    os.replace(tmp, output)
    partial.unlink()


def duration_seconds(path: Path) -> float | None:
    try:
        with av.open(str(path)) as container:
            return container.duration / 1_000_000 if container.duration else None
    except av.FFmpegError:
        return None


class Finalizer(QObject):
    """마무리를 백그라운드 스레드에서 돌린다. 앱이 도중에 꺼지면 partial이 남아 다음 실행 때 복구된다."""

    finished = Signal(Path, Path)  # partial, output
    failed = Signal(Path, str)
    progress = Signal(Path, float, object)  # partial, 처리한 초, 전체 초(None = 모름)

    def run(self, partial: Path, output: Path) -> None:
        threading.Thread(target=self._run, args=(partial, output), daemon=True).start()

    def _run(self, partial: Path, output: Path) -> None:
        try:
            finalize_recording(partial, output, lambda done, total: self.progress.emit(partial, done, total))
        except Exception as e:
            log.exception("마무리 실패: %s", partial)
            self.failed.emit(partial, str(e))
            return
        self.finished.emit(partial, output)
