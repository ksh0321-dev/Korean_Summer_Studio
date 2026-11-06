"""몸 모델 내려받기 (인터넷 없이 file:// 로) · 사람 상자 찾기."""

import zipfile

import numpy as np

from ks_studio.services.body import (
    BACKGROUND,
    CLOTHES,
    PARTS_FILE,
    PARTS_SIZE,
    POSE_FILE,
    body_models_ready,
    download_body_models,
    person_box,
)


def test_download_body_models_unzips_pose_and_leaves_no_temp_files(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    with zipfile.ZipFile(source / "pose.zip", "w") as zf:
        zf.writestr("rtmpose/end2end.onnx", b"pose-model")
        zf.writestr("rtmpose/pipeline.json", b"{}")
    (source / "parts.onnx").write_bytes(b"parts-model")
    folder = tmp_path / "models" / "body"
    progress = []
    assert not body_models_ready(folder)
    download_body_models(progress.append, folder, (source / "pose.zip").as_uri(), (source / "parts.onnx").as_uri())
    assert body_models_ready(folder)
    assert (folder / POSE_FILE).read_bytes() == b"pose-model"
    assert (folder / PARTS_FILE).read_bytes() == b"parts-model"
    assert sorted(p.name for p in folder.iterdir()) == sorted([POSE_FILE, PARTS_FILE])  # 임시 파일 없음
    assert progress and progress[-1] == 1.0 and progress == sorted(progress)


def test_person_box_picks_the_person_at_the_face():
    parts = np.zeros((PARTS_SIZE, PARTS_SIZE, 6), np.float16)
    parts[..., BACKGROUND] = 1
    for x0, x1 in ((10, 60), (150, 230)):  # 사람 둘 (오른쪽이 더 크다)
        parts[40:250, x0:x1, BACKGROUND] = 0
        parts[40:250, x0:x1, CLOTHES] = 1
    width, height = 1024, 2048  # 화면 크기 = 분리 결과의 4배·8배
    assert person_box(parts, width, height)[0] == 150 * 4  # 얼굴을 모르면 가장 큰 사람
    left = person_box(parts, width, height, near=(30 * 4, 100 * 8))
    assert left == (10 * 4, 40 * 8, 60 * 4, 250 * 8)  # 얼굴이 있는 사람
