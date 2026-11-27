import pytest

from scripts.serve_policy_rtc import resolve_fps, resolve_timing


def test_checkpoint_fps_is_authoritative():
    assert resolve_fps({"fps": 30}, None) == 30.0
    assert resolve_fps({"fps": 30}, 30.0) == 30.0
    with pytest.raises(ValueError, match="does not match"):
        resolve_fps({"fps": 30}, 15.0)
    with pytest.raises(ValueError, match="metadata"):
        resolve_fps({}, None)


def test_execution_fps_is_an_explicit_time_scale():
    assert resolve_timing({"fps": 30}, 30, 15) == (30.0, 15.0, 2.0)
    assert resolve_timing({"fps": 30}, None, None) == (30.0, 30.0, 1.0)
    with pytest.raises(ValueError, match="execution-fps"):
        resolve_timing({"fps": 30}, 30, 0)
