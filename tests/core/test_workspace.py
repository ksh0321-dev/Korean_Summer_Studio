from datetime import datetime

from ks_studio.core.workspace import Workspace, recording_name


def test_recording_name_is_date_time():
    assert recording_name(datetime(2026, 11, 3, 14, 25, 30)) == "20261103_142530"


def test_missing_folders_mean_no_files(tmp_path):
    ws = Workspace(tmp_path / "nothing")
    assert ws.recordings() == []
    assert ws.leftover_partials() == []


def test_recordings_newest_first_and_ignore_partials(tmp_path):
    ws = Workspace(tmp_path)
    ws.partial_dir.mkdir(parents=True)
    for name in ("20261103_100000", "20261103_120000"):
        ws.final_path(name).write_bytes(b"")
    ws.final_path("20261103_130000").with_name("20261103_130000.mp4.part").write_bytes(b"")
    ws.partial_path("20261103_140000").write_bytes(b"")

    assert [p.stem for p in ws.recordings()] == ["20261103_120000", "20261103_100000"]
    assert [p.stem for p in ws.leftover_partials()] == ["20261103_140000"]
