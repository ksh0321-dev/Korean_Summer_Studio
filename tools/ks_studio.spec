# PyInstaller 설정 — tools/build_exe.py가 부른다 (직접: pyinstaller tools/ks_studio.spec).
# 한 폴더(onedir) 배포: dist/KS_Studio-<버전>/KS Studio.exe + _internal/ (Python·Qt·FFmpeg·CUDA·cuDNN 포함)

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "src"))
from ks_studio import __version__  # noqa: E402

site = Path(sys.prefix) / "Lib" / "site-packages"


def tree(folder: Path, dest: str) -> list[tuple[str, str]]:
    """folder 안 DLL을 같은 상대 위치(dest/...)로. onnxruntime이 nvidia/<패키지>/bin 구조로 찾는다."""
    return [(str(f), str(Path(dest) / f.parent.relative_to(folder))) for f in folder.rglob("*.dll")]


binaries = []
binaries += tree(site / "nvidia" / "cu13" / "bin", "nvidia/cu13/bin")
binaries += tree(site / "nvidia" / "cudnn" / "bin", "nvidia/cudnn/bin")
# 음성 인식(ctranslate2 4.8)은 CUDA 12의 cuBLAS를 찾는다 (얼굴 모델의 onnxruntime은 CUDA 13).
# 실측: 이 PC는 CUDA 12.8 툴킷이 깔려 있어 소스로는 돌았지만, 깨끗한 환경의 exe에서는 cublas64_12.dll이 없었다
binaries += tree(site / "nvidia" / "cublas" / "bin", "nvidia/cublas/bin")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")

datas = []
datas += collect_data_files("ks_studio", includes=["ui/theme/*.qss", "assets/fonts/*", "assets/icon.*"])
datas += collect_data_files("faster_whisper")  # 말소리 구간 찾기 모델 (silero_vad)
datas += collect_data_files("insightface")

a = Analysis(
    [str(ROOT / "src" / "ks_studio" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=["insightface.app", "insightface.model_zoo", "insightface.utils.face_align", "skimage.transform"],
    runtime_hooks=[str(ROOT / "tools" / "pyi_rthook_cuda.py")],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KS Studio",
    icon=str(ROOT / "src" / "ks_studio" / "assets" / "icon.ico"),  # 탐색기·작업 표시줄의 exe 아이콘
    console=False,  # 터미널 창 없이 (로그는 user/logs/)
    upx=False,  # CUDA DLL을 압축하면 깨지거나 느려진다
)
coll = COLLECT(exe, a.binaries, a.datas, name=f"KS_Studio-{__version__}", upx=False)
