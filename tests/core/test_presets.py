"""AI 보정 프리셋: preset/<이름>.json 저장·불러오기·삭제, 예전·손으로 고친 파일."""

import json

from ks_studio.core.beauty import BeautySettings
from ks_studio.core.presets import apply_preset, delete_preset, list_presets, load_preset, save_preset, valid_name


def test_save_load_roundtrip_keeps_effects_only(tmp_path):
    folder = tmp_path / "preset"
    mine = BeautySettings(smooth=0.7, chin=-0.4, lip=0.6, eyeshadow=0.5, eyeshadow_color="coral", approved=True, target=(0.1, 0.2))
    save_preset(folder, "데일리", mine)
    assert list_presets(folder) == ["데일리"]
    data = json.loads((folder / "데일리.json").read_text(encoding="utf-8"))
    assert "target" not in data["effects"] and "approved" not in data["effects"]  # 사람·OK는 영상마다
    other = BeautySettings(blush=0.9, approved=True, target=(0.5,))
    loaded = apply_preset(other, load_preset(folder, "데일리"))
    assert (loaded.smooth, loaded.chin, loaded.lip, loaded.blush) == (0.7, -0.4, 0.6, 0.0)  # 프리셋에 없던 값은 0
    assert loaded.eyeshadow_color == "coral"  # 색도 프리셋에
    assert loaded.target == (0.5,) and not loaded.approved  # 사람은 그대로, OK는 다시


def test_old_or_hand_edited_preset_files(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"version": 1, "effects": {"blush": 1, "future": 3, "slim": "x"}}), encoding="utf-8")
    assert load_preset(tmp_path, "old") == {"blush": 1.0}  # 정수 받음, 모르는 칸·잘못된 값은 버림


def test_list_sorted_and_delete(tmp_path):
    for name in ("b", "A", "c"):
        save_preset(tmp_path, name, BeautySettings())
    assert list_presets(tmp_path) == ["A", "b", "c"]
    delete_preset(tmp_path, "b")
    assert list_presets(tmp_path) == ["A", "c"]
    assert list_presets(tmp_path / "없음") == []


def test_valid_names():
    assert valid_name("촬영용 진하게") and not valid_name("  ") and not valid_name("a/b") and not valid_name(".숨김")
    assert not valid_name("x" * 41)
