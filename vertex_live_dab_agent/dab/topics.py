"""DAB protocol topic names and payload contracts.

All topic templates follow the DAB 2.0 specification format:
  ``dab/<device_id>/<command-path>``

Use :func:`format_topic` to substitute ``{device_id}`` before publishing.
"""

# ---------------------------------------------------------------------------
# Topic templates  (use format_topic() to resolve {device_id})
# ---------------------------------------------------------------------------
TOPIC_APPLICATIONS_LAUNCH = "dab/{device_id}/applications/launch"
TOPIC_APPLICATIONS_GET_STATE = "dab/{device_id}/applications/get-state"
TOPIC_INPUT_KEY_PRESS = "dab/{device_id}/input/key-press"
TOPIC_OUTPUT_IMAGE = "dab/{device_id}/output/image"
TOPIC_DEVICE_INFO = "dab/{device_id}/device/info"
TOPIC_SYSTEM_RESTART = "dab/{device_id}/system/restart"

# ---------------------------------------------------------------------------
# Standard DAB key codes
# ---------------------------------------------------------------------------
KEY_UP = "KEY_UP"
KEY_DOWN = "KEY_DOWN"
KEY_LEFT = "KEY_LEFT"
KEY_RIGHT = "KEY_RIGHT"
KEY_OK = "KEY_ENTER"
KEY_BACK = "KEY_BACK"
KEY_HOME = "KEY_HOME"

# Mapping from planner ActionType strings to DAB key codes
KEY_MAP: dict[str, str] = {
    "PRESS_UP": KEY_UP,
    "PRESS_DOWN": KEY_DOWN,
    "PRESS_LEFT": KEY_LEFT,
    "PRESS_RIGHT": KEY_RIGHT,
    "PRESS_OK": KEY_OK,
    "PRESS_BACK": KEY_BACK,
    "PRESS_HOME": KEY_HOME,
}


def format_topic(template: str, device_id: str) -> str:
    """Substitute ``{device_id}`` in a topic template.

    Example::

        format_topic(TOPIC_INPUT_KEY_PRESS, "tv-living-room")
        # -> "dab/tv-living-room/input/key-press"
    """
    return template.format(device_id=device_id)
