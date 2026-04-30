# DAB Control Android App

This folder contains the Android frontend for the `vertex_live_dab_agent` API.

## Current status

- Compose app scaffold
- Hilt DI
- Retrofit + OkHttp + Kotlinx Serialization
- SharedPreferences API base URL setting
- Dashboard screen wired to:
  - `GET /health`
  - `GET /system/metrics`

## Open in Android Studio

1. Open this `android/` folder as a project.
2. Let Android Studio sync Gradle.
3. Set API URL in app (default: `http://10.99.57.66:8081`).
4. Run on emulator/device.

## Next

- Add full endpoint interfaces and repositories for runs, YTS, DAB, IR.
- Add navigation + feature screens to match web frontend parity.
