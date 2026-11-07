"""새 버전 알림: GitHub main의 __init__.py에서 __version__을 읽어 지금 버전과 비교한다. 순수 파이썬 — Qt import 금지.

저장소가 비공개거나 인터넷이 없으면 조용히 None (프로그램 쓰는 데 지장 없게).
"""

import logging
import re
import urllib.request

from ks_studio import __version__

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/ksh0321-dev/Korean_Summer_Studio"
VERSION_URL = "https://raw.githubusercontent.com/ksh0321-dev/Korean_Summer_Studio/main/src/ks_studio/__init__.py"
TIMEOUT = 5  # 초
_VERSION = re.compile(r'__version__\s*=\s*"([^"]+)"')


def parse_version(text: str) -> tuple[int, ...]:
    """'0.2.10' → (0, 2, 10). 숫자가 아닌 꼬리(-beta 등)는 버린다."""
    numbers = []
    for part in text.strip().split("."):
        digits = re.match(r"\d+", part)
        if not digits:
            break
        numbers.append(int(digits.group()))
    return tuple(numbers)


def newer_version(remote_source: str, current: str = __version__) -> str | None:
    """원격 __init__.py 내용에서 지금보다 새 버전이면 그 버전 문자열."""
    found = _VERSION.search(remote_source)
    if found and parse_version(found.group(1)) > parse_version(current):
        return found.group(1)
    return None


def check_for_update(url: str = VERSION_URL, timeout: float = TIMEOUT) -> str | None:
    """새 버전 문자열 또는 None (같거나, 확인할 수 없거나). 네트워크를 쓰니 백그라운드에서 부른다."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": f"KS-Studio/{__version__}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return newer_version(response.read(64_000).decode("utf-8", "replace"))
    except (OSError, ValueError) as e:  # URLError·HTTPError(비공개 저장소 404)·시간 초과는 OSError
        log.info("새 버전 확인 못 함: %s", e)
        return None
