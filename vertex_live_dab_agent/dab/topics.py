"""DAB protocol topic names and payload contracts."""

# Request topics
TOPIC_APPLICATIONS_LAUNCH = "dab/{device_id}/applications/launch"
TOPIC_APPLICATIONS_GET_STATE = "dab/{device_id}/applications/get-state"
TOPIC_INPUT_KEY_PRESS = "dab/{device_id}/input/key-press"
TOPIC_OUTPUT_IMAGE = "dab/{device_id}/output/image"
TOPIC_DEVICE_INFO = "dab/{device_id}/device/info"
TOPIC_SYSTEM_RESTART = "dab/{device_id}/system/restart"

# Key codes
KEY_UP = "KEY_UP"
KEY_DOWN = "KEY_DOWN"
KEY_LEFT = "KEY_LEFT"
KEY_RIGHT = "KEY_RIGHT"
KEY_OK = "KEY_ENTER"
KEY_BACK = "KEY_BACK"
KEY_HOME = "KEY_HOME"

KEY_MAP = {
    "PRESS_UP": KEY_UP,
    "PRESS_DOWN": KEY_DOWN,
    "PRESS_LEFT": KEY_LEFT,
    "PRESS_RIGHT": KEY_RIGHT,
    "PRESS_OK": KEY_OK,
    "PRESS_BACK": KEY_BACK,
    "PRESS_HOME": KEY_HOME,
}
