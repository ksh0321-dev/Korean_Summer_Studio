"""같은 사람 묶기·찾기: 가짜 인식값 (사람마다 방향이 다른 벡터 + 약간의 흔들림)."""

import numpy as np

from ks_studio.core.people import cluster, match

rng = np.random.default_rng(1)
PEOPLE = [v / np.linalg.norm(v) for v in rng.normal(size=(3, 512))]


def face_of(person: int, noise: float = 0.5) -> np.ndarray:
    v = PEOPLE[person] + rng.normal(scale=noise / np.sqrt(512), size=512)
    return v / np.linalg.norm(v)


def test_cluster_groups_same_person_across_scenes():
    faces = [face_of(p) for p in (0, 1, 0, 0, 2, 1, 0)]
    labels = cluster(faces)
    assert labels[0] == labels[2] == labels[3] == labels[6]
    assert labels[1] == labels[5] and len(set(labels)) == 3


def test_match_finds_target_or_nobody():
    target = face_of(0)
    assert match([face_of(1), face_of(0), face_of(2)], target) == 1
    assert match([face_of(1), face_of(2)], target) is None  # 주인공이 없는 장면
    assert match([], target) is None
