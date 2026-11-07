"""프로젝트 백업: 자동 저장할 때 project.json을 backups/ 폴더에 시각을 붙여 몇 개씩 남겨 둔다.

자동 저장은 같은 파일을 덮어쓰므로, 잘못 편집하고 저장된 뒤에는 되돌릴 곳이 없었다 (프로그램을 껐다 켜면 되돌리기 기록도 없다).
- 마지막 백업에서 BACKUP_EVERY초가 지났고 내용이 달라졌을 때만 남긴다 (2초마다 저장해도 파일이 쌓이지 않게)
- 프로젝트를 열 때 한 번 남긴다 (그날 처음 상태로 돌아갈 수 있게)
- KEEP개를 넘으면 오래된 것부터 지운다
"""

import shutil
from datetime import datetime
from pathlib import Path

BACKUP_EVERY = 300  # 초 (5분)
KEEP = 30  # 남겨 둘 백업 수 (5분마다면 2시간 반 남짓)
STAMP = "%Y%m%d_%H%M%S"


def backup_dir(project_file: Path) -> Path:
    return project_file.parent / "backups"


def list_backups(project_file: Path) -> list[tuple[datetime, Path]]:
    """[(시각, 파일)] 최신 순."""
    found = []
    for path in backup_dir(project_file).glob("project_*.json"):
        try:
            found.append((datetime.strptime(path.stem.removeprefix("project_"), STAMP), path))
        except ValueError:
            continue
    return sorted(found, reverse=True)


def make_backup(project_file: Path, now: datetime | None = None, force: bool = False) -> Path | None:
    """지금 project.json을 백업한다 (force가 아니면 BACKUP_EVERY초 안에 한 번, 내용이 같으면 건너뛴다). 남긴 파일 또는 None."""
    if not project_file.exists():
        return None
    now = now or datetime.now()
    existing = list_backups(project_file)
    if existing and not force:
        last_time, last_path = existing[0]
        if (now - last_time).total_seconds() < BACKUP_EVERY:
            return None
    if existing and existing[0][1].read_bytes() == project_file.read_bytes():
        return None
    folder = backup_dir(project_file)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"project_{now.strftime(STAMP)}.json"
    shutil.copy2(project_file, target)
    for _, old in list_backups(project_file)[KEEP:]:
        old.unlink(missing_ok=True)
    return target


def _loads(path: Path) -> bool:
    from ks_studio.core.project import Project  # 순환 import 없음

    try:
        Project.load(path)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def repair(project_file: Path, now: datetime | None = None) -> tuple[Path | None, Path | None]:
    """project.json이 읽히지 않으면 망가진 것을 backups/broken_*.json으로 치우고 읽히는 최신 백업으로 바꾼다.

    (되살린 백업, 치운 파일). 멀쩡하면 (None, None). 읽히는 백업이 없으면 빈 프로젝트로 시작한다 (망가진 파일은 남는다).
    """
    if not project_file.exists() or _loads(project_file):
        return None, None
    now = now or datetime.now()
    folder = backup_dir(project_file)
    folder.mkdir(parents=True, exist_ok=True)
    broken = folder / f"broken_{now.strftime(STAMP)}.json"
    shutil.move(project_file, broken)
    for _, path in list_backups(project_file):
        if _loads(path):
            shutil.copy2(path, project_file)
            return path, broken
    return None, broken


SKIP_ON_COPY = ("cache", "exports", "backups", ".test", ".partial")  # 다시 만들 수 있거나 복제본에 필요 없는 것


def duplicate_project(source_root: Path, target_root: Path) -> None:
    """프로젝트 폴더를 통째로 복사한다 (캐시·내보낸 영상·백업은 빼고). 프로젝트 안 경로는 상대 경로라 그대로 맞는다."""
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"비어 있지 않은 폴더입니다: {target_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, target_root, ignore=shutil.ignore_patterns(*SKIP_ON_COPY), dirs_exist_ok=True)


def copy_size(source_root: Path) -> int:
    """duplicate_project가 복사할 바이트 수."""
    total = 0
    for path in source_root.rglob("*"):
        if path.is_file() and not any(part in SKIP_ON_COPY for part in path.relative_to(source_root).parts):
            total += path.stat().st_size
    return total
