from vertex_live_dab_agent.system_ops.recovery_engine import detect_repeated_loop


def test_detects_capture_getstate_loop():
    actions = ["CAPTURE_SCREENSHOT", "GET_STATE", "CAPTURE_SCREENSHOT", "GET_STATE", "CAPTURE_SCREENSHOT"]
    assert detect_repeated_loop(actions) is True


def test_detects_back_capture_loop():
    actions = ["PRESS_BACK", "PRESS_BACK", "PRESS_BACK", "CAPTURE_SCREENSHOT", "CAPTURE_SCREENSHOT"]
    assert detect_repeated_loop(actions) is True


def test_ignores_normal_short_sequence():
    actions = ["LAUNCH_APP", "WAIT", "GET_STATE"]
    assert detect_repeated_loop(actions) is False
