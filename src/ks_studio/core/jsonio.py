import json
import os
from pathlib import Path


def write_json_atomic(path: Path, data: object) -> None:
    """임시 파일에 쓰고 교체 → 저장 중 꺼져도 기존 파일은 깨지지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
