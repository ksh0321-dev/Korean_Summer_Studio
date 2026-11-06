import pytest

from ks_studio.core.settings import AppSettings, DeviceRef, SyncCalibration, find_calibration, with_calibration
from ks_studio.core.sync import contact_time, estimate_delay_ms

JOYTRON = DeviceRef("usb#cam1", "JOYTRON HD20")
PHONE = DeviceRef("usb#vid_04e8", "SAMSUNG_Android")
MIC = DeviceRef("{mic}", "마이크(USB PnP)")
OTHER_MIC = DeviceRef("{mic2}", "Focusrite")


def test_delay_is_median_and_spread_is_range():
    # 실측 휴대폰 웹캠: 손이 닿은 시각이 소리보다 약 100ms 늦음
    assert estimate_delay_ms([0.100, 0.106, 0.098]) == (100, 8)
    with pytest.raises(ValueError):
        estimate_delay_ms([])


def test_contact_time_is_between_frames():
    assert contact_time(1.092, 1.109) == pytest.approx(1.1005)


def test_calibration_is_per_camera_and_mic_pair():
    joytron = SyncCalibration(JOYTRON, MIC, 10, "2026-11-05")
    phone = SyncCalibration(PHONE, MIC, 105, "2026-11-05")
    calibrations = [joytron, phone]
    assert find_calibration(calibrations, PHONE, MIC) is phone
    assert find_calibration(calibrations, PHONE, OTHER_MIC) is None  # 마이크가 바뀌면 다시 재야 한다
    assert find_calibration(calibrations, None, MIC) is None


def test_renamed_device_still_matches_by_id():
    """실측: 같은 휴대폰이 'SAMSUNG_Android'에서 'USB 비디오 장치'로 이름이 바뀌어 떴다."""
    calibrations = [SyncCalibration(PHONE, MIC, 105, "2026-11-05")]
    renamed = DeviceRef(PHONE.id, "USB 비디오 장치")
    assert find_calibration(calibrations, renamed, MIC).audio_delay_ms == 105


def test_new_measurement_replaces_old_for_same_pair():
    old = SyncCalibration(PHONE, MIC, 90, "2026-11-01")
    other = SyncCalibration(JOYTRON, MIC, 10, "2026-11-01")
    updated = with_calibration([old, other], SyncCalibration(PHONE, MIC, 105, "2026-11-05"))
    assert [c.audio_delay_ms for c in updated] == [10, 105]


def test_settings_round_trip_with_calibrations(tmp_path):
    settings = AppSettings(camera=PHONE, microphone=MIC, sync_calibrations=[SyncCalibration(PHONE, MIC, 105, "2026-11-05")])
    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)
    assert loaded == settings
    assert loaded.sync_calibration.audio_delay_ms == 105
