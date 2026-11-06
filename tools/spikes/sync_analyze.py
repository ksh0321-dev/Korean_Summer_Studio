"""싱크 측정: 박수 소리 위치를 자동으로 찾고, 그 앞뒤 영상 프레임을 한 장(콘택트 시트)으로 저장한다.

사용법: python sync_analyze.py <녹화.mp4> <출력폴더>

시트에서 손이 닿은 프레임을 찾으면
    오프셋 = 그 프레임의 dt (프레임 시각 - 소리 시각)
    음수 = 소리가 화면보다 늦음 / 양수 = 소리가 화면보다 먼저 나옴
"""

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from ks_studio.services.sync import clap_onsets

BEFORE_S, AFTER_S = 0.2, 0.25  # 소리 기준 앞뒤 시간 (fps와 상관없이 같은 범위)
SCALE = 0.4  # 가운데 9:16 영역을 이 비율로 축소


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    rate = 48000
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks), rate


def frames_around(path: Path, onset: float) -> list[tuple[float, Image.Image, float]]:
    """(프레임 시각, 가운데 9:16 축소 이미지, 직전 프레임과의 움직임 크기)"""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        lo, hi = onset - BEFORE_S, onset + AFTER_S
        container.seek(max(0, int((lo - 1.0) / stream.time_base)), stream=stream, backward=True)
        out, prev = [], None
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t > hi:
                break
            gray = frame.to_ndarray(format="gray")
            h, w = gray.shape
            crop_w = round(h * 9 / 16)
            x0 = (w - crop_w) // 2
            motion = float(np.abs(gray[::8, ::8].astype(np.int16) - prev).mean()) if prev is not None else 0.0
            prev = gray[::8, ::8].astype(np.int16)
            if t >= lo:
                image = frame.to_image().crop((x0, 0, x0 + crop_w, h))
                out.append((t, image.resize((round(crop_w * SCALE), round(h * SCALE))), motion))
        return out


def contact_sheet(frames, onset: float, title: str) -> Image.Image:
    tw, th = frames[0][1].size
    cols = 8
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 36) + 40), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill="white")
    for k, (t, image, motion) in enumerate(frames):
        x, y = (k % cols) * tw, 40 + (k // cols) * (th + 36)
        sheet.paste(image, (x, y))
        dt_ms = (t - onset) * 1000
        draw.text((x + 6, y + th + 4), f"t={t:.3f}s  dt={dt_ms:+.0f}ms  move={motion:.1f}", fill="yellow")
    return sheet


def main() -> None:
    path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, rate = load_audio(path)
    onsets = clap_onsets(samples, rate)
    print(f"{path.name}: 오디오 {len(samples) / rate:.2f}s, 박수 후보 {len(onsets)}개")
    for k, onset in enumerate(onsets, 1):
        frames = frames_around(path, onset)
        sheet_path = out_dir / f"{path.stem}_clap{k}_{onset:.3f}s.png"
        contact_sheet(frames, onset, f"{path.name}  clap {k}  sound onset {onset:.4f}s").save(sheet_path)
        print(f"  clap {k}: 소리 {onset:.4f}s -> {sheet_path.name}")


if __name__ == "__main__":
    main()
