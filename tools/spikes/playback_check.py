"""검증: 임시 내보내기 결과, 재생 위치가 실제 시간과 같은 속도로 가는지. 사용법: python playback_check.py <녹화.mp4> <출력폴더>"""

import sys
import threading
import time
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.export import export_project
from ks_studio.services.playback import Player
from ks_studio.services.sources import SourceCache

source, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
app = QApplication([])
ws = Workspace(source.parent.parent)
sources = SourceCache()
src = ws.to_source(source)
project = Project(clips=(Clip(src, 100, 130, id="A"), Clip(src, 10, 20, audio_offset_ms=50, id="B")))

# 1) 임시 내보내기
out = out_dir / "export_test.mp4"
t0 = time.perf_counter()
ok = export_project(project, ws, sources, out, (1920, 1080), lambda d, total: None, threading.Event())
print(f"export ok={ok} {time.perf_counter() - t0:.2f}s")
with av.open(str(out)) as c:
    v = c.streams.video[0]
    frames = [f.to_ndarray(format="gray").astype(np.int16) for f in c.decode(v)]
    print("export frames", len(frames), "audio streams", len(c.streams.audio), "duration", c.duration / 1e6)
with av.open(str(source)) as c:
    s = c.streams.video[0]
    truth = {round(f.pts * s.time_base * 30): f.to_ndarray(format="gray").astype(np.int16) for f in c.decode(s)}
for t, k in ((0, 100), (30, 10), (39, 19)):
    right = np.abs(frames[t] - truth[k]).mean()
    wrong = np.abs(frames[t] - truth[k + 5]).mean()
    print(f"  export t={t}: 원본 {k}와 차이 {right:.2f} (5프레임 옆과는 {wrong:.2f})")

# 2) 재생 시계: 위치 변화를 실제 시간과 비교
long_project = Project(clips=(Clip(src, 0, 390, id="L"),))
player = Player(sources)
player.set_project(long_project, ws)
log = []
player.position_changed.connect(lambda p: log.append((time.perf_counter(), p)))


def start() -> None:
    log.clear()
    player.play()


def report() -> None:
    player.pause()
    started = [r for r in log if r[1] > 0]
    t_first, p_first = started[0]
    t_last, p_last = started[-1]
    rate = (p_last - p_first) / (t_last - t_first)
    print(f"재생: {p_last - p_first}프레임 / {t_last - t_first:.2f}s → {rate:.2f}fps (기대 30)")
    print(f"첫 화면까지 {(t_first - t_play[0]) * 1000:.0f}ms")
    app.quit()


t_play = []
QTimer.singleShot(1500, lambda: (t_play.append(time.perf_counter()), start()))
QTimer.singleShot(7500, report)
app.exec()
