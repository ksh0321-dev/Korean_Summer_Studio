"""앱이 쓰는 폴더. 코드와 사용자 데이터를 나눈다 — 저장소·배포본에 개인 설정이 섞이지 않게.

APP_DIR     소스로 돌리면 저장소 폴더, exe면 exe가 있는 폴더. 모든 위치는 여기 기준 상대 위치라 폴더째 옮겨도 된다
USER_DIR    APP_DIR/user — 개인 설정(장치·싱크 보정·최근 프로젝트)·프리셋·기본 프로젝트. 저장소에 올리지 않고,
            새 버전 exe로 바꿀 때는 이 폴더만 옮기면 된다
MODELS_DIR  APP_DIR/models — 처음 쓸 때 내려받는 AI 모델 (크다, 저장소에 올리지 않음)
"""

import shutil
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))  # PyInstaller로 만든 exe
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[3]
USER_DIR = APP_DIR / "user"
SETTINGS_PATH = USER_DIR / "settings.json"
DEFAULT_WORKSPACE = USER_DIR / "workspace"  # 프로젝트를 고르지 않았을 때의 기본 프로젝트
PRESETS_DIR = USER_DIR / "presets"  # AI 보정 프리셋 (프리셋마다 .json 하나)
CAPTION_STYLES_DIR = USER_DIR / "caption_styles"  # 자막 모양 프리셋 (프리셋마다 .json 하나)
GLOSSARY_PATH = USER_DIR / "glossary.json"  # 자막 용어집 (앱 전체 공통)
PEOPLE_PATH = USER_DIR / "people.json"  # 등록한 사람 (얼굴을 흐리지 않을 사람, 앱 전체 공통)
MODELS_DIR = APP_DIR / "models"  # AI 얼굴·음성 인식 모델


def app_relative(path: Path) -> str:
    """설정에 적을 경로: APP_DIR 안이면 상대 경로(폴더째 옮겨도 맞다), 밖이면 절대 경로."""
    path = Path(path).resolve()
    try:
        return path.relative_to(APP_DIR).as_posix() or "."
    except ValueError:
        return str(path)


def from_app_relative(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else APP_DIR / path


def migrate_user_files() -> None:
    """예전 위치(저장소 바로 아래)의 개인 설정·프리셋을 user/로 옮긴다. 이미 옮겼으면 아무것도 안 한다."""
    moves = ((APP_DIR / "settings.json", SETTINGS_PATH), (APP_DIR / "preset", PRESETS_DIR))
    for old, new in moves:
        if old.exists() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
    _relative_project_paths()


def _relative_project_paths() -> None:
    """설정의 프로젝트 경로 중 앱 폴더 안의 것을 상대 경로로 (예전 설정은 절대 경로였다 — 같은 폴더가 두 번 보이지 않게)."""
    import json

    if not SETTINGS_PATH.exists():
        return
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    workspace = data.get("workspace_dir")
    recent = data.get("recent_projects", [])
    new_workspace = app_relative(from_app_relative(workspace)) if workspace else workspace
    new_recent = list(dict.fromkeys(app_relative(from_app_relative(p)) for p in recent))
    if (new_workspace, new_recent) != (workspace, recent):
        from ks_studio.core.jsonio import write_json_atomic

        write_json_atomic(SETTINGS_PATH, {**data, "workspace_dir": new_workspace, "recent_projects": new_recent})
