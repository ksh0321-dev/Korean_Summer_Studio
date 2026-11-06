"""녹화 페이지: 미리보기(반전) · 3-2-1 카운트다운 · 녹화/정지(Space) · 5초 테스트 · 박수 싱크 보정 · 녹화본 목록.

카메라는 한 곳에서만 열 수 있으므로 이 페이지가 보일 때만 켠다 (activate / deactivate).
"""

import logging
import threading
import time
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.settings import AppSettings, DeviceRef, SyncCalibration, with_calibration
from ks_studio.core.workspace import Workspace, recording_name
from ks_studio.services.capture import CameraPreview, MicInput
from ks_studio.services.devices import DeviceMonitor, best_format, find_camera, find_microphone, supported_fps
from ks_studio.services.media import Finalizer
from ks_studio.services.recording import Recorder
from ks_studio.services.sync import find_claps
from ks_studio.ui.recorder.calibration_dialog import CalibrationDialog
from ks_studio.ui.recorder.playback_dialog import PlaybackDialog
from ks_studio.ui.recorder.recordings_list import RecordingsList
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.status_bar import ERROR, INFO, LIVE, WARNING
from ks_studio.ui.widgets.level_meter import LevelMeter
from ks_studio.ui.widgets.video_view import VideoView

log = logging.getLogger(__name__)

COUNTDOWN_S = 3
TEST_SECONDS = 5
CALIBRATION_SECONDS = 12
MODE_RECORD, MODE_TEST, MODE_CALIBRATE = "record", "test", "calibrate"
AUTO_STOP_S = {MODE_TEST: TEST_SECONDS, MODE_CALIBRATE: CALIBRATION_SECONDS}
BADGES = {MODE_RECORD: "REC", MODE_TEST: "TEST", MODE_CALIBRATE: "SYNC"}


class _ClapFinder(QObject):
    found = Signal(object)  # list[Clap]

    def run(self, path: Path) -> None:
        threading.Thread(target=self._run, args=(path,), daemon=True, name="clap-finder").start()

    def _run(self, path: Path) -> None:
        try:
            claps = find_claps(path)
        except Exception:
            log.exception("박수 분석 실패: %s", path)
            claps = []
        self.found.emit(claps)


def _missing_text(label: str, ref: DeviceRef | None) -> str:
    if ref is None:
        return f"⚠ {label} 미설정 — ⚙ 설정에서 선택"
    return f"⚠ {label} 연결 안 됨 ({ref.name})"


class RecorderPage(QWidget):
    busy_changed = Signal(bool)  # 카운트다운·녹화 중이면 True → 단계 전환·설정 잠금
    settings_edited = Signal(object)  # fps·가이드 변경 → 저장
    idle = Signal()  # 녹화도 마무리도 남지 않음 → 앱을 닫아도 됨
    edit_requested = Signal(Path)  # 녹화본을 컷편집에 넣어 달라
    message = Signal(str, str)  # 상태 표시줄 알림 (글, 수준)
    indicator = Signal(str, str, str)  # 상태 표시줄 오른쪽 표시 (키, 글, 수준). 빈 글이면 숨김

    def __init__(self, devices: DeviceMonitor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = AppSettings()
        self._workspace: Workspace | None = None
        self._active = False
        self._camera_ok = False
        self._mic_ok = False
        self._fmt = None
        self._mode = MODE_RECORD
        self._take = 0
        self._current: tuple[Path, Path] | None = None  # 녹화 중 (partial, output)
        self._pending: dict[Path, Path] = {}  # 마무리 중 partial → output
        self._progress: dict[Path, tuple[float, float | None]] = {}  # partial → (처리한 초, 전체 초)
        self._processed_total = 0.0  # 지금까지 마무리한 미디어 초 (처리 속도 계산용)
        self._closing = False
        self._record_started = 0.0
        self._countdown_left = 0

        self.view = VideoView()
        self.camera = CameraPreview(self.view.sink, self)
        self.mic = MicInput(self)
        self.recorder = Recorder(self.view.sink, self.mic, self)
        self.finalizer = Finalizer(self)
        self._clap_finder = _ClapFinder(self)
        self._clap_finder.found.connect(self._show_calibration)
        self.meter = LevelMeter()
        self.meter.setMaximumWidth(96)

        self.fps_combo = QComboBox()
        self.fps_info = QLabel()
        self.fps_info.setWordWrap(True)
        self.guide_check = QCheckBox("9:16 세로 가이드")
        self.record_button = QPushButton()
        self.record_button.setObjectName("recordButton")
        self.test_button = QPushButton(f"{TEST_SECONDS}초 테스트 녹화")
        self.sync_label = QLabel()
        self.sync_label.setObjectName("syncStatus")
        self.sync_label.setWordWrap(True)
        self.calibrate_button = QPushButton("박수로 싱크 보정")
        self.recordings = RecordingsList()
        self.recordings.open_requested.connect(self.edit_requested)
        for widget in (self.record_button, self.test_button, self.calibrate_button, self.fps_combo, self.guide_check):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space가 버튼 클릭이 아니라 녹화 단축키로 가게

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(200)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        # 실제 카메라 fps: 프레임이 올 때마다 세고(카메라 스레드), 1초마다 상태 표시줄에 알린다
        self._frames = 0
        self._frames_since = time.monotonic()
        self._fps_warmup = True
        self.view.sink.videoFrameChanged.connect(self._count_frame, Qt.ConnectionType.DirectConnection)
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._report_fps)
        self._fps_timer.start()

        self.mic.level.connect(self.meter.set_peak)
        self.camera.error.connect(self._on_camera_error)
        self.recorder.stopped.connect(self._on_recorder_stopped)
        self.recorder.failed.connect(self._on_recorder_failed)
        self.finalizer.finished.connect(self._on_finalized)
        self.finalizer.failed.connect(self._on_finalize_failed)
        self.finalizer.progress.connect(self._on_finalize_progress)
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        self.guide_check.toggled.connect(self._on_guide_toggled)
        self.record_button.clicked.connect(self._toggle_record)
        self.test_button.clicked.connect(lambda: self._start_countdown(MODE_TEST))
        self.calibrate_button.clicked.connect(self._begin_calibration)
        devices.cameras_changed.connect(self._on_devices_changed)
        devices.microphones_changed.connect(self._on_devices_changed)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._toggle_record)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(self._cancel_countdown)

        controls = QWidget()
        form = QFormLayout()
        form.addRow("fps", self.fps_combo)
        column = QVBoxLayout(controls)
        column.addLayout(form)
        column.addWidget(self.fps_info)
        column.addWidget(self.guide_check)
        column.addSpacing(12)
        column.addWidget(self.record_button)
        column.addWidget(self.test_button)
        column.addSpacing(8)
        column.addWidget(self.sync_label)
        column.addWidget(self.calibrate_button)
        column.addWidget(self.meter, 1, Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(QLabel("마이크 레벨"), 0, Qt.AlignmentFlag.AlignHCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(stage_layout(center=self.view, right=controls, bottom=self.recordings))
        self._update_controls()

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, settings: AppSettings, workspace: Workspace) -> None:
        self._active = True
        self._settings = settings
        self._workspace = workspace
        self.guide_check.blockSignals(True)
        self.guide_check.setChecked(settings.show_vertical_guide)
        self.guide_check.blockSignals(False)
        self.view.set_guide(settings.show_vertical_guide)
        self._start_devices()
        self._refresh_recordings()

    def deactivate(self) -> None:
        self._active = False
        self.camera.stop()
        self.mic.stop()
        self._camera_ok = self._mic_ok = False
        self.view.set_message("")
        self.indicator.emit("fps", "", INFO)
        self._update_controls()

    def begin_close(self) -> bool:
        """앱을 닫기 직전: 녹화 중이면 멈추고 장치를 끈다. 마무리할 작업이 남았으면 True."""
        self._closing = True
        self._cancel_countdown()
        self._stop_recording()
        self.deactivate()
        return bool(self._pending)

    def finalize_progress(self) -> tuple[float, float | None]:
        """(지금까지 마무리한 미디어 초, 남은 미디어 초 — 아직 모르면 None)"""
        remaining = 0.0
        for partial in self._pending:
            done, total = self._progress.get(partial, (0.0, None))
            if total is None:
                return self._processed_total, None
            remaining += max(0.0, total - done)
        return self._processed_total, remaining

    def offer_recovery(self) -> None:
        """비정상 종료로 남은 녹화가 있으면 MP4로 복구할지 묻는다."""
        leftovers = self._workspace.leftover_partials() if self._workspace else []
        if not leftovers:
            return
        answer = QMessageBox.question(
            self,
            "녹화 복구",
            f"마무리되지 못한 녹화가 {len(leftovers)}개 있습니다 (비정상 종료).\n"
            "지금 MP4로 복구할까요?\n\n아니요를 누르면 다음 실행 때 다시 묻습니다.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for partial in leftovers:
            self._finalize(partial, self._workspace.final_path(partial.stem))
        self._refresh_recordings()

    # ── 장치 ────────────────────────────────────────────────────

    def _start_devices(self) -> None:
        camera = find_camera(self._settings.camera)
        mic = find_microphone(self._settings.microphone)

        fps_list = supported_fps(camera) if camera else []
        fps = self._settings.fps if self._settings.fps in fps_list else max(fps_list, default=None)
        self.fps_combo.blockSignals(True)
        self.fps_combo.clear()
        for value in fps_list:
            self.fps_combo.addItem(f"{value} fps", value)
        if fps is not None:
            self.fps_combo.setCurrentIndex(self.fps_combo.findData(fps))
        self.fps_combo.blockSignals(False)
        mismatch = fps is not None and fps != self._settings.fps
        self.fps_info.setText(f"⚠ 이 카메라는 1080p {self._settings.fps}fps 미지원 → {fps}fps" if mismatch else "")

        self._fmt = best_format(camera, fps) if fps else None
        self._camera_ok = False
        if camera is None:
            self.camera.stop()
            self.view.set_message(_missing_text("카메라", self._settings.camera))
        elif self._fmt is None:
            self.camera.stop()
            self.view.set_message("⚠ 이 카메라는 1080p 30/60fps를 지원하지 않습니다")
        else:
            self.view.set_message("카메라 여는 중…")
            self.camera.start(camera, self._fmt)
            self._fps_warmup = True  # 켜지는 중인 첫 1초는 프레임이 덜 와서 fps가 낮게 잡힌다
            self.recorder.reset(fps)
            self._camera_ok = True

        self._mic_ok = mic is not None
        if mic is None:
            self.mic.stop()
            self._say(_missing_text("마이크", self._settings.microphone), WARNING)
        else:
            self.mic.start(mic)
        self._update_sync_label()
        self._update_controls()

    def _on_devices_changed(self) -> None:
        if not self._active:
            return
        if self.recorder.is_recording:
            if find_camera(self._settings.camera) is None or find_microphone(self._settings.microphone) is None:
                self._stop_recording()
                self._say("⚠ 녹화 중 장치 연결이 끊겨 녹화를 멈췄습니다. 그때까지 분량은 저장합니다.", ERROR)
            return
        self._cancel_countdown()
        self._start_devices()

    def _on_camera_error(self, message: str) -> None:
        self._camera_ok = False
        if self.recorder.is_recording:
            self._stop_recording()
            self._say(f"⚠ 카메라 오류로 녹화를 멈췄습니다: {message}", ERROR)
        self.view.set_message(f"카메라를 열 수 없음: {message}\n다른 앱이 카메라를 쓰고 있는지 확인")
        self._update_controls()

    # ── 녹화 흐름: 대기 → 카운트다운 → 녹화 → 마무리 ─────────────

    def _ready(self) -> bool:
        return (
            self._active
            and self._camera_ok
            and self._mic_ok
            and not self.recorder.is_recording
            and not self._countdown_timer.isActive()
        )

    def _toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_recording()
        elif self._countdown_timer.isActive():
            self._cancel_countdown()
        else:
            self._start_countdown(MODE_RECORD)

    def _start_countdown(self, mode: str) -> None:
        if not self._ready():
            return
        self._mode = mode
        self._countdown_left = COUNTDOWN_S
        self.view.set_overlay(str(COUNTDOWN_S))
        self._countdown_timer.start()
        self.busy_changed.emit(True)
        self._update_controls()

    def _tick_countdown(self) -> None:
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.view.set_overlay(str(self._countdown_left))
            return
        self._countdown_timer.stop()
        self.view.set_overlay("")
        self._begin_recording()

    def _cancel_countdown(self) -> None:
        if not self._countdown_timer.isActive():
            return
        self._countdown_timer.stop()
        self.view.set_overlay("")
        self.busy_changed.emit(False)
        self._update_controls()

    def _begin_recording(self) -> None:
        ws = self._workspace
        if self._mode == MODE_TEST:
            partial, output = ws.test_partial, ws.test_output
        elif self._mode == MODE_CALIBRATE:
            partial, output = ws.calibration_partial, ws.calibration_output
        else:
            name = recording_name(datetime.now())
            partial, output = ws.partial_path(name), ws.final_path(name)
        partial.parent.mkdir(parents=True, exist_ok=True)
        resolution = self._fmt.resolution()
        # 보정용 녹화는 보정 없이 찍어야 실제 지연을 잴 수 있다
        delay = 0 if self._mode == MODE_CALIBRATE else self._audio_delay_ms()
        self.recorder.start(partial, self.fps_combo.currentData(), (resolution.width(), resolution.height()), delay)
        self._current = (partial, output)
        self._take += 1
        self._record_started = time.monotonic()
        self._elapsed_timer.start()
        self._update_elapsed()
        if self._mode == MODE_CALIBRATE:
            self._say(f"손 옆면이 보이게 박수 3번 (2초 간격) — {CALIBRATION_SECONDS}초 뒤 자동으로 멈춥니다", WARNING)
        else:
            self._say("녹화를 시작했습니다 — Space로 정지")
        if self._mode in AUTO_STOP_S:
            QTimer.singleShot(AUTO_STOP_S[self._mode] * 1000, lambda take=self._take: take == self._take and self._stop_recording())
        self._update_controls()

    def _stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        self.recorder.stop()  # 쓰기 스레드가 MKV를 닫으면 stopped → 마무리 시작
        partial, output = self._current
        self._current = None
        self._pending[partial] = output
        self._elapsed_timer.stop()
        self.view.set_badge("")
        self.indicator.emit("recording", "", INFO)
        self._say(f"마무리 중: {output.name}")
        self.busy_changed.emit(False)
        self._update_controls()
        self._refresh_recordings()

    def _update_elapsed(self) -> None:
        seconds = int(time.monotonic() - self._record_started)
        badge = f"● {BADGES[self._mode]} {seconds // 60:02d}:{seconds % 60:02d}"
        self.view.set_badge(badge)
        self.indicator.emit("recording", badge, LIVE)

    def _on_recorder_stopped(self, partial: Path) -> None:
        output = self._pending.get(partial)
        if output is not None:
            self.finalizer.run(partial, output)

    def _on_recorder_failed(self, message: str) -> None:
        if self._current is not None:
            self._pending.pop(self._current[0], None)
        self._current = None
        self._elapsed_timer.stop()
        self.view.set_badge("")
        self.indicator.emit("recording", "", INFO)
        self._say(f"⚠ 녹화 실패: {message}", ERROR)
        self.busy_changed.emit(False)
        self._update_controls()
        self._emit_if_idle()

    def _finalize(self, partial: Path, output: Path) -> None:
        self._pending[partial] = output
        self.finalizer.run(partial, output)

    def _on_finalize_progress(self, partial: Path, done: float, total: float | None) -> None:
        previous = self._progress.get(partial, (0.0, None))[0]
        self._processed_total += max(0.0, done - previous)
        self._progress[partial] = (done, total)

    def _on_finalized(self, partial: Path, output: Path) -> None:
        self._finish_job(partial)
        self._refresh_recordings()
        self._update_controls()
        if self._workspace and output == self._workspace.test_output:
            self._say("테스트 녹화 완료")
            if self._active and not self._closing:
                dialog = PlaybackDialog(output, self)
                dialog.exec()
                dialog.deleteLater()
        elif self._workspace and output == self._workspace.calibration_output:
            if self._active and not self._closing:
                self._finish_calibration(output)
        else:
            self._say(f"저장됨: {output.name}")
        self._emit_if_idle()

    def _on_finalize_failed(self, partial: Path, message: str) -> None:
        self._finish_job(partial)
        self._say(f"⚠ 마무리 실패 ({partial.name}): {message} — 원본은 {partial.parent}에 남아 있습니다.", ERROR)
        self._refresh_recordings()
        self._update_controls()
        self._emit_if_idle()

    def _finish_job(self, partial: Path) -> None:
        self._pending.pop(partial, None)
        done, total = self._progress.pop(partial, (0.0, None))
        if total is not None:
            self._processed_total += max(0.0, total - done)

    def _emit_if_idle(self) -> None:
        if not self._pending and not self.recorder.is_recording:
            self.idle.emit()

    # ── 설정·표시 ───────────────────────────────────────────────

    def _say(self, text: str, level: str = INFO) -> None:
        self.message.emit(text, level)

    def _count_frame(self, _frame) -> None:
        """카메라 스레드에서 불린다. 세기만 한다."""
        self._frames += 1

    def _report_fps(self) -> None:
        now = time.monotonic()
        frames, elapsed = self._frames, now - self._frames_since
        self._frames, self._frames_since = 0, now
        if not (self._active and self._camera_ok) or elapsed <= 0:
            return
        if self._fps_warmup:
            self._fps_warmup = False
            return
        fps = frames / elapsed
        target = self._fmt.maxFrameRate() if self._fmt is not None else self.fps_combo.currentData()
        if fps < 0.9 * target:  # 어두우면 카메라가 스스로 fps를 낮춘다 (실측 JOYTRON)
            self.indicator.emit("fps", f"카메라 {fps:.1f}fps ⚠ 설정 {target:g}fps보다 낮음 (조명 확인)", WARNING)
        else:
            self.indicator.emit("fps", f"카메라 {fps:.1f}fps", INFO)

    def _on_fps_changed(self) -> None:
        self._settings = replace(self._settings, fps=self.fps_combo.currentData())
        self.settings_edited.emit(self._settings)
        self._start_devices()

    def _on_guide_toggled(self, checked: bool) -> None:
        self.view.set_guide(checked)
        self._settings = replace(self._settings, show_vertical_guide=checked)
        self.settings_edited.emit(self._settings)

    def _update_controls(self) -> None:
        recording = self.recorder.is_recording
        busy = recording or self._countdown_timer.isActive()
        self.record_button.setText("■ 정지  (Space)" if recording else "● 녹화  (Space)")
        self.record_button.setProperty("recording", recording)
        self.record_button.style().polish(self.record_button)
        self.record_button.setEnabled(recording or self._ready())
        test_pending = self._workspace is not None and self._workspace.test_partial in self._pending
        self.test_button.setEnabled(self._ready() and not test_pending)
        calibration_pending = self._workspace is not None and self._workspace.calibration_partial in self._pending
        self.calibrate_button.setEnabled(self._ready() and not calibration_pending)
        self.fps_combo.setEnabled(not busy)

    def _refresh_recordings(self) -> None:
        if self._workspace is None:
            return
        hidden = (self._workspace.test_output, self._workspace.calibration_output)
        pending = [out.name for out in self._pending.values() if out not in hidden]
        self.recordings.show_recordings(self._workspace.recordings_dir, self._workspace.recordings(), pending)

    # ── 박수 싱크 보정 ──────────────────────────────────────────

    def _audio_delay_ms(self) -> int:
        calibration = self._settings.sync_calibration
        return calibration.audio_delay_ms if calibration else 0

    def _update_sync_label(self) -> None:
        calibration = self._settings.sync_calibration
        if calibration is None:
            self.sync_label.setText("⚠ 이 카메라·마이크 조합은 아직 싱크 보정 전입니다. 박수로 보정하세요.")
        else:
            self.sync_label.setText(f"싱크 보정: 오디오 {calibration.audio_delay_ms:+d}ms ({calibration.measured_at} 측정)")
        self.sync_label.setProperty("warning", calibration is None)
        self.sync_label.style().polish(self.sync_label)

    def _begin_calibration(self) -> None:
        answer = QMessageBox.information(
            self,
            "박수로 싱크 보정",
            f"카운트다운이 끝나면 {CALIBRATION_SECONDS}초 동안 녹화합니다.\n\n"
            "· 평소 촬영 위치에 앉고, 마이크도 평소 위치에 두세요\n"
            "· 손 옆면이 카메라에 보이게, 화면 가운데에서 크게 박수 3번 (2초 간격)\n"
            "· 조명을 밝게 (어두우면 카메라가 fps를 낮춥니다)",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Ok:
            self._start_countdown(MODE_CALIBRATE)

    def _finish_calibration(self, output: Path) -> None:
        """박수 분석은 몇 초 걸려서 백그라운드에서 한다 (메인 스레드가 멈추면 마이크 입력이 밀린다)."""
        self._say("박수를 찾는 중…")
        self._clap_finder.run(output)

    def _show_calibration(self, claps: list) -> None:
        if not self._active or self._closing:
            return
        camera, mic = self._settings.camera, self._settings.microphone
        dialog = CalibrationDialog(claps, camera.name, mic.name, self)
        accepted = dialog.exec() == CalibrationDialog.DialogCode.Accepted
        delay = dialog.delay_ms
        dialog.deleteLater()
        if not accepted:
            self._say("싱크 보정을 취소했습니다")
            return
        calibration = SyncCalibration(camera, mic, delay, date.today().isoformat())
        self._settings = replace(
            self._settings, sync_calibrations=with_calibration(self._settings.sync_calibrations, calibration)
        )
        self.settings_edited.emit(self._settings)
        self._update_sync_label()
        self._say(f"싱크 보정 저장: 오디오 {delay:+d}ms — 이후 녹화에 적용됩니다")
