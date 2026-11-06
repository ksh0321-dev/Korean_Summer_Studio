"""실험: InsightFace 얼굴 찾기 + 106점 랜드마크. GPU 속도, 점 번호 배치(볼·코 위치)를 그림으로 확인.

사용법: python face_spike.py <출력폴더> <영상>...
"""

import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

started = time.monotonic()
app = FaceAnalysis(
    name="buffalo_l",
    root=str(ROOT / "models"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_2d_106"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
print(f"모델 준비 {time.monotonic() - started:.1f}s · 모듈 {list(app.models)}", flush=True)
for name, model in app.models.items():
    print(f"  {name}: {[p for p in model.session.get_providers()]}", flush=True)

for arg in sys.argv[2:]:
    path = Path(arg)
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="bgr24") for _, f in zip(range(40), c.decode(video=0))]
    image = frames[-1]
    app.get(image)  # 예열
    t = time.perf_counter()
    for f in frames[:20]:
        faces = app.get(f)
    per = (time.perf_counter() - t) / 20 * 1000
    faces = app.get(image)
    print(f"\n{path.name}: {image.shape[1]}x{image.shape[0]} · 얼굴 {len(faces)}개 · 프레임당 {per:.1f}ms", flush=True)
    for face in faces:
        lm = face.landmark_2d_106
        print(f"  box {np.round(face.bbox).astype(int)} score {face.det_score:.2f} · landmark {lm.shape}", flush=True)
        canvas = image.copy()
        for i, (x, y) in enumerate(lm.astype(int)):
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(canvas, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        x0, y0, x1, y1 = np.round(face.bbox).astype(int)
        pad = int((x1 - x0) * 0.3)
        crop = canvas[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        scale = 900 / max(crop.shape[:2])
        cv2.imwrite(str(out / f"{path.stem}_landmarks.png"), cv2.resize(crop, None, fx=scale, fy=scale))
