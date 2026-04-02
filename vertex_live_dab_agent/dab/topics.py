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
TOPIC_APPLICATIONS_LIST = "dab/{device_id}/applications/list"
TOPIC_APPLICATIONS_EXIT = "dab/{device_id}/applications/exit"
TOPIC_INPUT_KEY_PRESS = "dab/{device_id}/input/key-press"
TOPIC_INPUT_LONG_KEY_PRESS = "dab/{device_id}/input/long-key-press"
TOPIC_INPUT_KEY_LIST = "dab/{device_id}/input/key/list"
TOPIC_OUTPUT_IMAGE = "dab/{device_id}/output/image"
TOPIC_OPERATIONS_LIST = "dab/{device_id}/operations/list"
TOPIC_DEVICE_INFO = "dab/{device_id}/device/info"
TOPIC_SYSTEM_RESTART = "dab/{device_id}/system/restart"
TOPIC_SYSTEM_SETTINGS_LIST = "dab/{device_id}/system/settings/list"
TOPIC_SYSTEM_SETTINGS_GET = "dab/{device_id}/system/settings/get"
TOPIC_SYSTEM_SETTINGS_SET = "dab/{device_id}/system/settings/set"
TOPIC_CONTENT_OPEN = "dab/{device_id}/content/open"
TOPIC_VOICE_LIST = "dab/{device_id}/voice/list"

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
    "PRESS_MENU": "KEY_MENU",
    "PRESS_EXIT": "KEY_EXIT",
    "PRESS_INFO": "KEY_INFO",
    "PRESS_GUIDE": "KEY_GUIDE",
    "PRESS_CAPTIONS": "KEY_CAPTIONS",
    "PRESS_PAGE_UP": "KEY_PAGE_UP",
    "PRESS_PAGE_DOWN": "KEY_PAGE_DOWN",
    "PRESS_PLAY_PAUSE": "KEY_PLAY_PAUSE",
    "PRESS_PLAY": "KEY_PLAY",
    "PRESS_PAUSE": "KEY_PAUSE",
    "PRESS_RECORD": "KEY_RECORD",
    "PRESS_STOP": "KEY_STOP",
    "PRESS_FAST_FORWARD": "KEY_FASTFORWARD",
    "PRESS_REWIND": "KEY_REWIND",
    "PRESS_SKIP_REWIND": "KEY_SKIP_REWIND",
    "PRESS_SKIP_FAST_FORWARD": "KEY_SKIP_FAST_FORWARD",
    "PRESS_NEXT": "KEY_NEXT",
    "PRESS_PREVIOUS": "KEY_PREVIOUS",
    "PRESS_VOLUME_UP": "KEY_VOLUMEUP",
    "PRESS_VOLUME_DOWN": "KEY_VOLUMEDOWN",
    "PRESS_MUTE": "KEY_MUTE",
    "PRESS_CHANNEL_UP": "KEY_CHANNELUP",
    "PRESS_CHANNEL_DOWN": "KEY_CHANNELDOWN",
    "PRESS_POWER": "KEY_POWER",
    "PRESS_RED": "KEY_RED",
    "PRESS_GREEN": "KEY_GREEN",
    "PRESS_YELLOW": "KEY_YELLOW",
    "PRESS_BLUE": "KEY_BLUE",
    "PRESS_0": "KEY_0",
    "PRESS_1": "KEY_1",
    "PRESS_2": "KEY_2",
    "PRESS_3": "KEY_3",
    "PRESS_4": "KEY_4",
    "PRESS_5": "KEY_5",
    "PRESS_6": "KEY_6",
    "PRESS_7": "KEY_7",
    "PRESS_8": "KEY_8",
    "PRESS_9": "KEY_9",
}


def format_topic(template: str, device_id: str) -> str:
    """Substitute ``{device_id}`` in a topic template.

    Example::

        format_topic(TOPIC_INPUT_KEY_PRESS, "tv-living-room")
        # -> "dab/tv-living-room/input/key-press"
    """
    return template.format(device_id=device_id)
