"""배포용 exe 만들기: Python·CUDA가 없는 Windows PC에서도 폴더째 복사해서 바로 실행.

    python tools\\build_exe.py                 dist\\KS_Studio-<버전>\\ 를 만든다
    python tools\\build_exe.py --with-models   이 PC의 models\\(얼굴·음성 인식 모델)도 넣는다 — 인터넷 없는 PC용, +3GB

버전은 src/ks_studio/__init__.py의 __version__ 하나 (폴더 이름·창 제목이 따른다).
결과 폴더 구조 (모두 상대 경로라 어디에 두어도 된다):
    KS Studio.exe     실행 파일
    _internal\\        Python·Qt·FFmpeg·CUDA·cuDNN·글꼴
    models\\           AI 모델 (없으면 처음 쓸 때 내려받는다)
    user\\             개인 설정·프리셋·기본 프로젝트·로그 (처음 켤 때 생긴다 — 새 버전으로 바꿀 때 이 폴더를 옮긴다)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ks_studio import __version__  # noqa: E402

GUIDE = """KS Studio {version}

실행: KS Studio.exe
- Python·CUDA 설치 없이 돈다. NVIDIA 그래픽 카드(드라이버)가 있으면 AI 보정·음성 인식·인코딩이 GPU로, 없으면 CPU로 돈다.
- AI 모델은 처음 쓸 때 models\\ 폴더에 내려받는다 (얼굴 약 300MB, 음성 인식 약 3GB, 인터넷 필요).
- 개인 설정·프리셋·기본 프로젝트·로그는 user\\ 폴더에 생긴다.

새 버전으로 바꿀 때: 새 폴더에 예전 폴더의 user\\ (와 받아 둔 models\\)를 복사한다.
문제가 생기면: user\\logs\\ks_studio.log 를 보내 주세요.
"""


def folder_size(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-models", action="store_true", help="models\\ 폴더도 함께 넣는다 (+약 3GB)")
    parser.add_argument("--zip", action="store_true", help="dist 폴더에 KS_Studio-<버전>.zip 으로도 묶는다 (전달용)")
    args = parser.parse_args()

    started = time.monotonic()
    out = ROOT / "dist" / f"KS_Studio-{__version__}"
    command = [sys.executable, "-m", "PyInstaller", str(ROOT / "tools" / "ks_studio.spec"), "--noconfirm",
               "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build")]
    # conda 파이썬은 자기 DLL(ffi·lzma·bz2·expat)을 Library/bin에 둔다 — conda activate 없이 돌리면 PATH에 없어
    # PyInstaller가 못 찾고, 그러면 exe가 다른 PC에서 _ctypes를 못 연다 (실측: 첫 빌드 경고) → 직접 넣어 준다
    env = dict(os.environ)
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if library_bin.exists():
        env["PATH"] = os.pathsep.join([str(library_bin), env.get("PATH", "")])
    print("빌드:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=env)
    (out / "사용법.txt").write_text(GUIDE.format(version=__version__), encoding="utf-8")
    if args.with_models and (ROOT / "models").exists():
        shutil.copytree(ROOT / "models", out / "models", dirs_exist_ok=True)
    print(f"\n완성: {out}  ({folder_size(out):.1f}GB, {time.monotonic() - started:.0f}초)", flush=True)
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"묶음: {archive}  ({Path(archive).stat().st_size / 1e9:.1f}GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
