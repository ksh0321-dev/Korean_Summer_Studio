from datetime import datetime, timedelta

from ks_studio.core.backups import (
    KEEP,
    copy_size,
    duplicate_project,
    list_backups,
    make_backup,
    repair,
)
from ks_studio.core.project import Project


def _project(tmp_path, fps=30):
    path = tmp_path / "project.json"
    Project(fps=fps).save(path)
    return path


def test_make_backup_throttles_and_skips_same_content(tmp_path):
    path = _project(tmp_path)
    t0 = datetime(2026, 11, 7, 12, 0, 0)
    assert make_backup(path, now=t0) is not None
    Project(fps=60).save(path)
    assert make_backup(path, now=t0 + timedelta(seconds=60)) is None  # 5분 안
    assert make_backup(path, now=t0 + timedelta(minutes=6)) is not None
    assert make_backup(path, now=t0 + timedelta(minutes=20)) is None  # 내용이 같다
    assert make_backup(path, now=t0 + timedelta(minutes=21), force=True) is None  # force여도 같으면 건너뛴다
    assert len(list_backups(path)) == 2


def test_make_backup_prunes_old(tmp_path):
    path = _project(tmp_path)
    t0 = datetime(2026, 11, 7, 12, 0, 0)
    for i in range(KEEP + 5):
        Project(fps=24 + i).save(path)
        make_backup(path, now=t0 + timedelta(minutes=10 * i))
    backups = list_backups(path)
    assert len(backups) == KEEP
    assert backups[0][0] == t0 + timedelta(minutes=10 * (KEEP + 4))  # 최신 순


def test_repair_restores_latest_loadable_backup(tmp_path):
    path = _project(tmp_path, fps=25)
    t0 = datetime(2026, 11, 7, 12, 0, 0)
    make_backup(path, now=t0)
    assert repair(path) == (None, None)  # 멀쩡하면 그대로
    path.write_text("{ 망가짐", encoding="utf-8")
    make_backup(path, now=t0 + timedelta(minutes=10))  # 망가진 백업은 건너뛴다
    restored, broken = repair(path, now=t0 + timedelta(minutes=11))
    assert restored is not None and broken.exists()
    assert Project.load(path).fps == 25


def test_repair_without_backup_starts_empty(tmp_path):
    path = tmp_path / "project.json"
    path.write_text("[]", encoding="utf-8")
    restored, broken = repair(path)
    assert restored is None and broken.exists() and not path.exists()


def test_duplicate_project_skips_derivatives(tmp_path):
    source = tmp_path / "a"
    _project(source.mkdir() or source)
    (source / "media").mkdir()
    (source / "media" / "clip.mp4").write_bytes(b"x" * 10)
    for skip in ("cache", "exports", "backups"):
        (source / skip).mkdir()
        (source / skip / "big.bin").write_bytes(b"y" * 1000)
    assert copy_size(source) == 10 + (source / "project.json").stat().st_size
    target = tmp_path / "b"
    duplicate_project(source, target)
    assert (target / "project.json").exists() and (target / "media" / "clip.mp4").exists()
    assert not (target / "cache").exists() and not (target / "exports").exists()
    try:
        duplicate_project(source, target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("비어 있지 않은 폴더에 복제하면 안 된다")
