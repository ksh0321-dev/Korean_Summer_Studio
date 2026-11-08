"""색 형식 바꾸기가 화면 밝기를 바꾸지 않는지 (0.3.1: 버림 대신 반올림 · 오갈 때 같은 색 규격) · 최고 화질 인코더."""

from fractions import Fraction

import av
import numpy as np
import pytest

from ks_studio.services.deliver import BEST_ENCODERS, ENCODERS, OutputFormat, encoders_for, fit_filters
from ks_studio.services.yuv import convert, from_rgb, matrix, to_rgb


def natural_frame(width=1280, height=720, colorspace=None) -> av.VideoFrame:
    """그러데이션 + 잡음 (실제 영상처럼 여러 밝기·색)."""
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    rgb = np.clip(np.stack([x, y, (x + y) / 2], axis=2) * 220 + 20 + rng.normal(0, 8, (height, width, 3)), 0, 255)
    frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgb24").reformat(format="yuv420p")
    if colorspace is not None:
        frame.colorspace = colorspace
    return frame


def luma(frame: av.VideoFrame) -> np.ndarray:
    return frame.to_ndarray()[: frame.height].astype(float)


@pytest.mark.parametrize("colorspace,height", [(None, 720), (1, 720), (None, 360)])
def test_rgb_roundtrip_keeps_brightness(colorspace, height):
    frame = natural_frame(height * 16 // 9, height, colorspace)
    back = from_rgb(to_rgb(frame), "yuv420p")
    assert abs(float((luma(back) - luma(frame)).mean())) < 0.05  # 예전(PyAV 기본): -1.1 · 709 표시 영상은 +1.4


def test_scaling_keeps_matrix_of_tagged_frames():
    frame = natural_frame(colorspace=1)  # BT.709 표시
    scaled = convert(frame, "yuv420p", 640, 360)
    plain = convert(natural_frame(), "yuv420p", 640, 360)
    assert abs(float((luma(scaled) - luma(plain)).mean())) < 0.05  # 표시가 있어도 규격을 바꿔 다시 계산하지 않는다
    graph = av.filter.Graph()  # 내보내기의 크기 맞추기 필터도 같다
    src = graph.add_buffer(width=1280, height=720, format="yuv420p", time_base=Fraction(1, 30))
    graph.link_nodes(src, *(graph.add(n, a) for n, a in fit_filters((1280, 720), (1280, 720), "crop")),
                     graph.add("buffersink")).configure()
    frame.pts = 0
    graph.push(frame)
    assert abs(float((luma(graph.pull()) - luma(frame)).mean())) < 0.05  # 예전: -1.1


def test_matrix_rule():
    assert matrix(1, 360) == "bt709" and matrix(6, 1080) == "bt601"
    assert matrix(None, 1080) == "bt709" and matrix(2, 480) == "bt601"  # 표시 없음: HD는 709, SD는 601


def test_best_quality_option():
    assert encoders_for(OutputFormat()) == ENCODERS["h264"]
    best = OutputFormat(codec="hevc", quality="best")
    assert encoders_for(best) == BEST_ENCODERS["hevc"] and best.tag.endswith("_최고화질")
    for codec in ENCODERS:  # 최고 화질은 같은 인코더의 화질 목표가 더 높다 (cq·crf가 더 낮다)
        normal, high = dict(ENCODERS[codec]), dict(BEST_ENCODERS[codec])
        nv = f"{codec}_nvenc" if codec == "h264" else "hevc_nvenc"
        assert int(high[nv]["cq"]) < int(normal[nv]["cq"])
