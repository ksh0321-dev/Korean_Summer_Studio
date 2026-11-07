"""용어집: 자주 나오는 한국어 낱말과 그 영어 (곶감 = dried persimmon). 앱 전체 공통 — user/glossary.json.

- 음성 인식: 한국어 낱말을 힌트로 줘서 받아쓰기가 그 낱말로 나오게 한다 (faster-whisper hotwords)
- 두 줄 자막 번역: 그 줄에 들어 있는 낱말의 영어를 힌트로 준다 (곶감 → dried persimmon)
- 앞으로 단어 강조(색칠)에도 쓴다
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ks_studio.core.jsonio import write_json_atomic

GLOSSARY_VERSION = 1
HINT_MAX = 60  # 음성 인식 힌트로 줄 낱말 수 (너무 많으면 힌트가 엉뚱한 곳에 끼어든다)


@dataclass(frozen=True)
class Term:
    korean: str
    english: str = ""


def load_glossary(path: Path) -> tuple[Term, ...]:
    if not path.exists():
        return ()
    try:
        items = json.loads(path.read_text(encoding="utf-8")).get("terms", [])
    except (OSError, ValueError):
        return ()
    terms = (Term(str(t.get("korean", "")).strip(), str(t.get("english", "")).strip()) for t in items if isinstance(t, dict))
    return tuple(t for t in terms if t.korean)


def save_glossary(path: Path, terms) -> None:
    clean = [{"korean": t.korean.strip(), "english": t.english.strip()} for t in terms if t.korean.strip()]
    write_json_atomic(path, {"version": GLOSSARY_VERSION, "terms": clean})


def speech_hint(terms) -> str | None:
    """받아쓰기 힌트: 한국어 낱말들 (없으면 None)."""
    words = [t.korean for t in terms][:HINT_MAX]
    return " ".join(words) if words else None


def translation_hint(terms, text: str) -> str | None:
    """번역 힌트: 이 줄에 나오는 낱말의 영어들 (없으면 None)."""
    found = [t.english for t in terms if t.english and t.korean in text]
    return ", ".join(found) if found else None
