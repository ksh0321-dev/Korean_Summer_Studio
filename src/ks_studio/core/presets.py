"""AI 보정 프리셋: 효과 값 묶음에 이름을 붙여 preset/<이름>.json 하나씩 둔다 (앱 전체 공통, 파일만 옮기면 다른 PC에서도).

담는 것은 효과 값(세기·색)뿐 — 보정할 사람·OK 상태는 영상마다 다르다.
"""

import json
from dataclasses import replace
from pathlib import Path

from ks_studio.core.beauty import PRESET_FIELDS, BeautySettings
from ks_studio.core.jsonio import write_json_atomic

PRESET_VERSION = 1
NAME_MAX = 40
INVALID_CHARS = set('\\/:*?"<>|')  # Windows 파일 이름에 못 쓰는 글자


def valid_name(name: str) -> bool:
    name = name.strip()
    return 0 < len(name) <= NAME_MAX and not (set(name) & INVALID_CHARS) and not name.startswith(".")


def list_presets(folder: Path) -> list[str]:
    return sorted((p.stem for p in folder.glob("*.json")), key=str.casefold) if folder.exists() else []


def save_preset(folder: Path, name: str, settings: BeautySettings) -> None:
    data = {"version": PRESET_VERSION, "effects": {f: getattr(settings, f) for f in PRESET_FIELDS}}
    write_json_atomic(folder / f"{name.strip()}.json", data)


def load_preset(folder: Path, name: str) -> dict:
    """프리셋의 효과 값. 모르는 칸은 버리고, 없는 칸(예전 프리셋에 없던 효과)은 넣지 않는다."""
    effects = json.loads((folder / f"{name}.json").read_text(encoding="utf-8")).get("effects", {})
    defaults = BeautySettings()
    values = {}
    for field in PRESET_FIELDS:
        default, value = getattr(defaults, field), effects.get(field)
        if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
            values[field] = float(value)  # 손으로 고친 파일의 0·1 같은 정수도 받는다
        elif isinstance(default, str) and isinstance(value, str):
            values[field] = value
    return values


def apply_preset(settings: BeautySettings, values: dict) -> BeautySettings:
    """프리셋을 입힌 설정: 프리셋에 없는 효과는 0(기본값), 보정할 사람은 그대로, OK는 풀린다 (샘플을 다시 보고 OK)."""
    defaults = BeautySettings()
    return replace(settings, **{f: values.get(f, getattr(defaults, f)) for f in PRESET_FIELDS}, approved=False)


def delete_preset(folder: Path, name: str) -> None:
    (folder / f"{name}.json").unlink(missing_ok=True)
