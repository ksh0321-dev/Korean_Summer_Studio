"""실험: PyAV로 1080p30 NV12 프레임을 h264_nvenc 인코딩. 사용법: python pyav_nvenc.py <출력.mkv>"""

import sys
import time
from fractions import Fraction

import av
import numpy as np

out = av.open(sys.argv[1], "w")
stream = out.add_stream("h264_nvenc", rate=30, options={"preset": "p5", "rc": "vbr", "b": "25M", "g": "30", "bf": "0"})
stream.width, stream.height, stream.pix_fmt = 1920, 1080, "nv12"
stream.time_base = Fraction(1, 30)

frames = 300
start = time.perf_counter()
for i in range(frames):
    buf = np.full((1080 * 3 // 2, 1920), i % 256, dtype=np.uint8)
    frame = av.VideoFrame.from_numpy_buffer(buf, format="nv12")
    frame.pts = i
    out.mux(stream.encode(frame))
out.mux(stream.encode())
out.close()
elapsed = time.perf_counter() - start
print(f"encoder={stream.codec_context.name} frames={frames} elapsed={elapsed:.2f}s -> {frames / elapsed:.0f} fps")
