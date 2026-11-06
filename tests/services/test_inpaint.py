"""배경 메꾸기: 패치는 메울 곳에만, 기억·앞 프레임 재사용, 카메라가 움직이면 기억 버리기, 모델 내려받기 (모델 없이 가짜 실행으로)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from ks_studio.services import inpaint  # noqa: E402
from ks_studio.services.inpaint import PATCH, FILL_MODELS, MemoryFill, PatchFill, download_fill_model, fill_model_ready  # noqa: E402
from ks_studio.services.reshape import GRID  # noqa: E402

H, W = 600, 800


def fake_model(calls):
    def run(windows, masks):
        calls.append(len(windows))
        assert all(w.shape == (PATCH, PATCH, 3) for w in windows) and all(m.shape == (PATCH, PATCH) for m in masks)
        return [np.full((PATCH, PATCH, 3), 200, np.uint8) for _ in windows]  # 메운 곳을 200으로
    return run


def scene():
    image = np.full((H, W, 3), 50, np.uint8)
    hole = np.zeros((H, W), bool)
    hole[200:400, 300:500] = True
    need = np.zeros((H, W), bool)
    need[200:400, 300:310] = True  # 사람이 비켜난 띠
    carry = np.zeros((H // GRID + 1, W // GRID + 1), np.float32)
    carry[200 // GRID : 400 // GRID, 300 // GRID : 500 // GRID] = 1  # 사람 자리
    return image, hole, need, carry


def test_patch_fill_runs_only_patches_covering_need():
    image, hole, need, carry = scene()
    calls = []
    fill = PatchFill("migan", run=fake_model(calls))
    box = (250, 150, 550, 450)
    out = fill(image, box, hole[150:450, 250:550], need[150:450, 250:550], carry)
    assert out.shape == (300, 300, 3)
    assert sum(calls) == fill.patches == 1  # 띠가 작아 패치 하나
    band = need[150:450, 250:550]
    assert (out[band] == 200).all()  # 메울 곳은 모델 값


def test_patch_fill_skips_when_nothing_to_fill():
    image, hole, _, carry = scene()
    calls = []
    out = PatchFill("lama", run=fake_model(calls))(image, (0, 0, W, H), hole, np.zeros((H, W), bool), carry)
    assert not calls and (out == image).all()


def test_memory_reuses_previous_fill_between_model_frames():
    image, hole, need, carry = scene()
    calls = []
    memory = MemoryFill(PatchFill("migan", run=fake_model(calls)))
    box = (0, 0, W, H)
    first = memory(image, box, hole, need, carry)
    assert (first[need] == 200).all() and len(calls) == 1
    second = memory(image, box, hole, need, carry)  # 같은 자리 → 모델 없이 앞 값
    assert len(calls) == 1 and (second[need] == 200).all()
    memory(image, box, hole, need, carry)
    memory(image, box, hole, need, carry)  # MODEL_EVERY(3)장마다 다시 모델 — 앞 값과 섞는다
    assert len(calls) == 2 and memory.model_calls == 2


def test_memory_uses_background_seen_before():
    image, hole, need, carry = scene()
    memory = MemoryFill(PatchFill("migan", run=fake_model([])))
    empty = np.zeros_like(carry)  # 첫 프레임: 사람이 없어 그 자리 배경(50)이 보인다
    memory(image, (0, 0, W, H), hole, np.zeros((H, W), bool), empty)
    out = memory(image, (0, 0, W, H), hole, need, carry)  # 다음 프레임: 사람이 비켜난 띠 = 아까 본 배경
    assert (out[need] == 50).all() and memory.remembered == need.sum()


def test_memory_forgets_when_camera_moves():
    image, hole, need, carry = scene()
    memory = MemoryFill(PatchFill("migan", run=fake_model([])))
    empty = np.zeros_like(carry)
    memory(image, (0, 0, W, H), hole, np.zeros((H, W), bool), empty)
    moved = np.full_like(image, 120)  # 배경이 통째로 바뀌었다 = 카메라가 움직였다
    for _ in range(MemoryFill.UPDATE_EVERY - 1):
        memory(moved, (0, 0, W, H), hole, np.zeros((H, W), bool), carry)
    out = memory(moved, (0, 0, W, H), hole, need, carry)
    assert memory.resets == 1 and (out[need] == 200).all()  # 옛 배경(50)을 쓰지 않는다


def test_download_fill_model(tmp_path):
    source = tmp_path / "src.onnx"
    source.write_bytes(b"model")
    folder = tmp_path / "inpaint"
    seen = []
    assert not fill_model_ready("lama", folder)
    download_fill_model("lama", seen.append, folder, source.as_uri())
    assert fill_model_ready("lama", folder) and seen[-1] == 1.0
    assert [p.name for p in folder.iterdir()] == [FILL_MODELS["lama"][1]]


def test_make_fill_falls_back_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(inpaint, "INPAINT_DIR", tmp_path)
    monkeypatch.setattr(inpaint, "fill_model_ready", lambda kind, folder=tmp_path: False)
    assert inpaint.make_fill("migan") is None and inpaint.make_fill("telea") is None
