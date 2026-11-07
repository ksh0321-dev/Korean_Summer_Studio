from ks_studio.services.updates import check_for_update, newer_version, parse_version


def test_parse_version():
    assert parse_version("0.2.10") == (0, 2, 10)
    assert parse_version("1.0.0-beta") == (1, 0, 0)
    assert parse_version("0.10") > parse_version("0.9.9")


def test_newer_version():
    assert newer_version('__version__ = "0.3.0"\n', "0.2.0") == "0.3.0"
    assert newer_version('__version__ = "0.2.0"\n', "0.2.0") is None
    assert newer_version('__version__ = "0.1.9"\n', "0.2.0") is None
    assert newer_version("엉뚱한 내용", "0.2.0") is None


def test_check_for_update_fails_quietly():
    assert check_for_update("http://127.0.0.1:9/none", timeout=0.5) is None
