# Camera Streaming Optimizations

## Issues Fixed
1. **Device Busy Handling** - Fast recovery from device busy errors
2. **Slow Stream Initialization** - Reduced startup latency
3. **High Latency** - Optimized for <100ms lag
4. **Stream Failures** - Faster detection and recovery

## Changes Made

### 1. Reduced Cooldown Periods (capture.py)
- `_hdmi_reprobe_interval_s`: 3.0s → 0.5s
  - Detects device changes/recovery faster
- `_hdmi_stream_reset_after_misses`: 20 → 8
  - Triggers reset sooner on failures
- `_hdmi_reset_cooldown_s`: 5.0s → 1.0s
  - Faster recovery from reset
- `_hdmi_release_grace_s`: 0.75s → 0.1s
  - Faster device release for quick re-opening
- `_busy_video_quarantine_s`: 30.0s → 3.0s
  - Aggressive recovery from device busy state

### 2. Reduced Frame Caching (hdmi_capture.py)
- `_max_cached_frame_age_s`: 2.0s → 0.05s (50ms)
  - Aggressively discard stale frames
  - Ensures < 100ms latency with fresh frames

### 3. Improved Reader Loop (hdmi_capture.py)
- `_max_consecutive_failures`: 6 (was hardcoded 12)
  - Detect failures sooner
- Reader sleep: 0.04s/0.08s → 0.01s/0.02s
  - Reduced CPU polling interval
  - Faster frame delivery
- Added `_consecutive_read_failures` tracking
  - Monitor frame delivery health

### 4. Optimized Frame Reading (hdmi_capture.py)
- `read_frame()` timeout: 0.45s → 0.1s
  - Non-blocking operation for responsiveness
- Always return latest frame available
  - Accept slightly stale frames vs. blocking
  - Better for real-time streaming

### 5. Android UI / Scrcpy ADB Streaming Fixes (scrcpy_stream.py)
- Prevented synchronous stream blocking when ADB device is dead/offline.
- Replaced the hard 4.0s synchronous fallback block with a non-blocking 0.45s wait that yields to the worker thread. 
- Reduced `adb screencap` internal timeout from 4.0s to 1.5s to fail-fast and auto-recover the connection quickly.
- Resolves the specific 5-second connection hang during stream startup when using Android UI capture mode.

## Expected Behavior

### Before Optimization
- Device busy errors → 30s wait
- Stream initialization → 2-5 seconds
- Android UI (ADB offline) startup delay → 5+ seconds
- Latency → 200-500ms
- Reset after error → 5+ seconds

### After Optimization
- Device busy errors → 3s recovery
- Stream initialization → 0.5-1.0 second
- Android UI (ADB offline) startup delay → fails fast / recovers in < 1 second
- Latency → 50-100ms
- Reset after error → 1-2 seconds

## Environment Variables

Override defaults with:
```bash
# Device busy quarantine (seconds)
CAMERA_BUSY_QUARANTINE_SECONDS=3.0

# Suppress noisy JPEG warnings
SUPPRESS_NATIVE_JPEG_WARNINGS=true

# Device holder recovery policy
CAMERA_BUSY_RECOVERY_POLICY=priority  # or "force"

# Force kill camera holders
FORCE_KILL_CAMERA_HOLDERS=true

# Process priority mapping
CAMERA_HOLDER_PRIORITIES=ffmpeg=10,python=40

# Device owner priority
CAMERA_DEVICE_OWNER_PRIORITY=60
```

## Latency Breakdown

With optimized settings, expected latency:
- Frame capture: 0-30ms (HDMI/V4L2)
- Reader thread processing: 10-20ms
- Application read: 0-20ms (no blocking)
- **Total: 50-100ms**

## Android Features Added

### YTS Report Analysis Tab
- Browse past result artifacts
- Select multiple results
- Run AI-driven triage analysis
- View deep analysis reports

### API Integration
- `/yts/results/artifacts` - List all artifacts
- `/yts/results/analyze` - Trigger analysis
- `/yts/results/analysis/{id}/txt` - Download analysis

### UI Components
- `AnalysisTab` - Artifact selection & analysis UI
- `YtsResultArtifactItemDto` - Artifact model
- `YtsResultsAnalysisRequestDto` - Request model
