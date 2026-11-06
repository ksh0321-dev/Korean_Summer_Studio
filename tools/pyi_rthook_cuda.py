"""exe가 켜질 때 가장 먼저: 함께 넣은 CUDA·cuDNN DLL 폴더를 DLL 검색 경로에 넣는다 (PyInstaller 런타임 훅).

CUDA를 설치하지 않은 PC에서도 GPU를 쓰게 하려고, pip의 nvidia 패키지(cu13·cudnn)를 exe 폴더에 그대로 넣었다.
onnxruntime(얼굴)·ctranslate2(음성 인식)는 이 DLL을 이름으로 찾으므로 os.add_dll_directory와 PATH 둘 다에 넣는다
(DLL이 다른 DLL을 부를 때는 PATH를 본다). NVIDIA 그래픽 드라이버는 PC에 있어야 한다 — GPU가 없으면 CPU로 돈다.
"""

import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
_dirs = [
    os.path.join(_base, "nvidia", "cu13", "bin", "x86_64"),
    os.path.join(_base, "nvidia", "cudnn", "bin"),
    os.path.join(_base, "nvidia", "cublas", "bin"),  # CUDA 12 cuBLAS (음성 인식 ctranslate2용)
    os.path.join(_base, "ctranslate2"),
    os.path.join(_base, "onnxruntime", "capi"),
]
for _d in _dirs:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
os.environ["PATH"] = os.pathsep.join([d for d in _dirs if os.path.isdir(d)] + [os.environ.get("PATH", "")])
