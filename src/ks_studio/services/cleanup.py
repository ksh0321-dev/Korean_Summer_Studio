"""프로젝트 안에 쌓이는 파생 파일 정리 — 원본(녹화본·가져온 영상)은 절대 지우지 않는다.

파생 파일은 모두 <프로젝트>/cache/ 아래에 있어 프로젝트 폴더를 지우면 같이 사라진다:
  cache/<키>.audio.npy · .peaks.npy · .thumbs2.npz      소리·파형·썸네일 (키 = sources.cache_name: 이름+경로·크기·시각 해시)
  cache/<키>.faces_<사람>.npy                            AI 보정 얼굴 분석 (영상 길이에 비례 — 20분 4K 60fps ≈ 60MB)
  cache/optimized/<키>.mp4                               편집용 사본 (원본만큼 크다)
  cache/ai_samples/<키>.v1.npz · cache/ai_clips/<키>.mp4  AI 샘플 후보 장면 · 1분 미리보기 영상
  cache/export_preview.mp4                               내보내기 저해상도 미리보기 (매번 덮어쓴다)
쓰다 만 파일(.part — 렌더 중 앱이 꺼진 것)과 저장 중 끊긴 .tmp도 남는다.

sweep: 프로젝트를 열 때 한 번. 지금 쓰는 영상(live)에 딸리지 않은 파생 파일, 예전 형식 캐시, 쓰다 만 파일을 지운다.
원본이 바뀌면(크기·시각) 키가 바뀌므로 예전 키의 캐시도 여기서 정리된다.
"""

import logging
import time
from collections.abc import Iterable
from pathlib import Path

from ks_studio.core.beauty import load_ai_state
from ks_studio.core.project import Project
from ks_studio.core.subtitles import load_subtitles
from ks_studio.core.workspace import Workspace
from ks_studio.services.face import track_file
from ks_studio.services.sources import OPTIMIZED_DIR, cache_name, optimized_path

log = logging.getLogger(__name__)

CURRENT_SUFFIXES = (".audio.npy", ".peaks.npy", ".thumbs2.npz")  # cache/ 바로 아래 파일 중 지금 쓰는 형식 (+ .faces_*.npy)
SUBDIRS = ("ai_samples", "ai_clips")
LEFTOVER_PATTERNS = ("*.part", "*.part.npz", "*.tmp")
PREVIEW_FILES = ("export_preview.mp4",)
BUSY_SECONDS = 120  # 이만큼 안에 바뀐 쓰다 만 파일은 지금 쓰는 중일 수 있다 (같은 프로젝트를 다시 열 때) → 남긴다


def _keys(cache_dir: Path, path: Path) -> set[str]:
    """path(원본)에 딸린 캐시 키: 원본 키 + 편집용 사본이 있으면 사본 키 (사본의 파형·썸네일은 사본 키로 저장된다)."""
    if not path.exists():
        return set()
    keys = {cache_name(path)}
    copy = optimized_path(cache_dir, path)
    if copy.exists():
        keys.add(cache_name(copy))
    return keys


def _owner(name: str, keys: set[str]) -> bool:
    return any(name == k or name.startswith(k + ".") for k in keys)


def derived_files(cache_dir: Path, path: Path) -> list[Path]:
    """원본 path에서 만든 파일 전부 (사본·캐시·AI 샘플·미리보기 영상). 원본은 들어 있지 않다."""
    keys = _keys(cache_dir, path)
    if not keys or not cache_dir.exists():
        return []
    found = [f for f in cache_dir.iterdir() if f.is_file() and _owner(f.name, keys)]
    for sub in SUBDIRS:
        folder = cache_dir / sub
        if folder.exists():
            found += [f for f in folder.iterdir() if f.is_file() and _owner(f.name, keys)]
    copy = optimized_path(cache_dir, path)
    if copy.exists():
        found.append(copy)
    return found


def size_of(files: Iterable[Path]) -> int:
    return sum(f.stat().st_size for f in files if f.exists())


def remove_files(files: Iterable[Path]) -> tuple[int, list[Path]]:
    """(지운 바이트 수, 못 지운 파일). Windows는 열려 있는 파일을 못 지운다 (실측: 사본을 만든 직후 썸네일을 만드느라
    읽고 있었다) → 못 지운 파일은 부른 쪽이 조금 뒤 다시 지우거나, 다음에 프로젝트를 열 때 sweep이 지운다."""
    freed, failed = 0, []
    for f in files:
        try:
            size = f.stat().st_size
            f.unlink()
            freed += size
        except FileNotFoundError:
            continue
        except OSError as e:
            log.info("지금은 못 지움 (나중에 다시): %s (%s)", f, e)
            failed.append(f)
    return freed, failed


def sweep(workspace: Workspace, live: Iterable[Path], keep: Iterable[Path] = ()) -> int:
    """쓰지 않는 파생 파일·쓰다 만 파일을 지운다. 지운 바이트 수.

    live = 지금 프로젝트가 쓰는 원본들. keep = 꼭 남길 파일 (지금 보정할 사람의 얼굴 분석처럼, 같은 영상에 여러 개가
    생기는 것 중 쓰는 것). 얼굴 분석 파일은 keep에 없으면 지운다 (사람을 바꾸면 예전 분석이 남는다).
    """
    cache = workspace.cache_dir
    keep = {Path(p).resolve() for p in keep}
    live = [p for p in live if p.exists()]
    keys = set().union(*(_keys(cache, p) for p in live)) if live else set()
    copies = {optimized_path(cache, p).name for p in live}
    doomed: list[Path] = [workspace.root / "project.tmp"]  # 프로젝트 폴더 바로 아래는 우리가 만든 것만 (사용자 파일이 있을 수 있다)
    for folder in (cache, cache / OPTIMIZED_DIR, *(cache / s for s in SUBDIRS), workspace.ai_dir,
                   workspace.exports_dir, workspace.final_dir, workspace.subtitles_file.parent):
        if folder.exists():
            for pattern in LEFTOVER_PATTERNS:
                doomed += [f for f in folder.glob(pattern) if time.time() - f.stat().st_mtime > BUSY_SECONDS]
    if cache.exists():
        for f in cache.iterdir():
            if not f.is_file() or f.name.endswith((".part", ".part.npz", ".tmp")):
                continue  # 쓰다 만 파일은 위에서 담았다
            if f.name in PREVIEW_FILES:
                doomed.append(f)
            elif ".faces_" in f.name:
                if f.resolve() not in keep:
                    doomed.append(f)
            elif not (_owner(f.name, keys) and f.name.endswith(CURRENT_SUFFIXES)):
                doomed.append(f)  # 쓰지 않는 영상의 캐시, 또는 예전 형식(thumbs.npz 등)
        optimized = cache / OPTIMIZED_DIR
        if optimized.exists():
            doomed += [f for f in optimized.glob("*.mp4") if f.name not in copies]  # 예전 형식 사본도 여기서
        for sub in SUBDIRS:
            folder = cache / sub
            if folder.exists():
                doomed += [f for f in folder.iterdir() if f.is_file() and not _owner(f.name, keys)]
    doomed = [f for f in doomed if f.exists()]
    freed, _ = remove_files(dict.fromkeys(doomed))
    if freed:
        log.info("프로젝트 정리: %.0fMB 지움", freed / 1e6)
    return freed


def project_files(workspace: Workspace) -> tuple[list[Path], set[Path]]:
    """(지금 프로젝트가 쓰는 원본들, 남길 얼굴 분석 파일). 탭마다 저장한 내용을 모두 본다.

    원본 = 워크스페이스 영상(녹화본·컷편집 결과·AI 보정본·가져온 것, 뺀 녹화본 제외) + 타임라인 클립 + AI·자막 탭 영상.
    """
    project = Project.load(workspace.project_file)
    ai = load_ai_state(workspace.ai_settings_file)
    subtitles = load_subtitles(workspace.subtitles_file)
    sources = set(workspace.videos((*ai.imported, *subtitles.imported))) | set(ai.settings) | set(subtitles.docs)
    sources |= {c.source for c in project.clips} | {o.clip.source for o in project.overlays}
    live = [workspace.resolve(s) for s in sources]
    keep = set()
    for source, settings in ai.settings.items():
        path = workspace.resolve(source)
        if path.exists():
            copy = optimized_path(workspace.cache_dir, path)
            media = copy if copy.exists() else path
            keep.add(track_file(workspace.cache_dir, cache_name(media), settings.target_vector))
    return live, keep


def clean_project(workspace: Workspace) -> int:
    """프로젝트를 열 때: 쓰지 않는 파생 파일을 지운다. 지운 바이트 수."""
    live, keep = project_files(workspace)
    return sweep(workspace, live, keep)
