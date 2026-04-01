"""System operation routing and device detection helpers."""

from .device_detection import (
	get_device_connection_type,
	get_device_platform_info,
	is_android_device,
	is_android_tv,
)
from .capabilities import (
	build_capability_snapshot,
	has_key,
	has_operation,
	has_setting,
	normalize_key_code,
	normalize_setting_key,
	normalize_setting_value,
	validate_setting_value,
)

