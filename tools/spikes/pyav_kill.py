"""실험: PyAV로 MKV(영상 NVENC + 오디오 PCM)를 실시간 속도로 쓰다가 강제 종료당했을 때 살아남는지. 사용법: python pyav_kill.py <출력.mkv>"""

import sys
import time
from fractions import Fraction

import av
import numpy as np

out = av.open(sys.argv[1], "w")
video = out.add_stream("h264_nvenc", rate=30, options={"preset": "p5", "b": "25M", "g": "30", "bf": "0"})
video.width, video.height, video.pix_fmt = 1920, 1080, "nv12"
video.time_base = Fraction(1, 30)
audio = out.add_stream("pcm_s16le", rate=48000, layout="mono")

start = time.perf_counter()
for i in range(30 * 60):
    buf = np.full((1080 * 3 // 2, 1920), i % 256, dtype=np.uint8)
    frame = av.VideoFrame.from_numpy_buffer(buf, format="nv12")
    frame.pts = i
    out.mux(video.encode(frame))

    samples = (np.sin(np.arange(1600) / 10) * 8000).astype(np.int16).reshape(1, -1)
    aframe = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    aframe.sample_rate, aframe.pts = 48000, i * 1600
    out.mux(audio.encode(aframe))

    time.sleep(max(0.0, start + (i + 1) / 30 - time.perf_counter()))
