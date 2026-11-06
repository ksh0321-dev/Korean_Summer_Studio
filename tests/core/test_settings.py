from ks_studio.core.settings import RECENT_LIMIT, AppSettings, DeviceRef, match_device, with_recent_project

CAM = DeviceRef(id="usb#port1#cam", name="JOYTRON HD20")
MIC = DeviceRef(id="{mic-1}", name="Analogue 1 + 2(Focusrite USB Audio)")


def test_missing_file_gives_empty_settings(tmp_path):
    assert AppSettings.load(tmp_path / "none.json") == AppSettings()


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "settings.json"
    settings = AppSettings(camera=CAM, microphone=MIC, workspace_dir="D:/ks", fps=60, show_vertical_guide=False)
    settings.save(path)
    assert AppSettings.load(path) == settings


def test_load_old_file_fills_new_fields_with_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera": null, "microphone": null}', encoding="utf-8")
    assert AppSettings.load(path) == AppSettings()


def test_recent_projects_most_recent_first_without_duplicates():
    recent = with_recent_project(["b", "a", "c"], "a")
    assert recent == ["a", "b", "c"]
    many = [str(i) for i in range(20)]
    assert len(with_recent_project(many, "new")) == RECENT_LIMIT


def test_match_prefers_id():
    same_name_other_id = DeviceRef(id="usb#port2#cam", name=CAM.name)
    assert match_device([same_name_other_id, CAM], CAM) == 1


def test_match_falls_back_to_name_when_port_changed():
    moved = DeviceRef(id="usb#port2#cam", name=CAM.name)
    assert match_device([MIC, moved], CAM) == 1


def test_no_match_returns_none():
    assert match_device([MIC], CAM) is None
