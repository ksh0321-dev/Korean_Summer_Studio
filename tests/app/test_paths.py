"""앱 폴더 기준 상대 경로: 앱 폴더 안은 상대, 밖은 절대."""

from pathlib import Path

from ks_studio.app.paths import APP_DIR, app_relative, from_app_relative


def test_paths_inside_app_dir_are_relative(tmp_path):
    assert app_relative(APP_DIR / "user" / "workspace") == "user/workspace"
    assert app_relative(APP_DIR) == "."
    assert from_app_relative("user/workspace") == APP_DIR / "user" / "workspace"
    outside = tmp_path / "proj"
    assert Path(app_relative(outside)) == outside.resolve() and from_app_relative(str(outside)) == outside
