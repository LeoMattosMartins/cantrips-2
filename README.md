# Gestureify

> Control Spotify with your hands. No keyboard. No mouse. Just gestures.

Gestureify is a local computer-vision application that reads your webcam feed in real time and maps hand gestures to Spotify playback commands. The CV pipeline runs entirely on your machine using [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and [JAX](https://jax.readthedocs.io/), with no cloud inference and no perceptible latency. Configuration is validated at startup by [Pydantic v2](https://docs.pydantic.dev/latest/) — a bad `.env` file produces a structured error listing every problem at once, not a cryptic crash three layers deep.

---

## Gestures

| Gesture | Action |
| :--- | :--- |
| Hold open palm for ~1.5 s | Toggle session **ON** / **OFF** |
| Closed fist *(session active)* | Pause playback |
| Swipe hand right *(session active)* | Next track |
| Swipe hand left *(session active)* | Previous track |
| Pinch thumb + index *(session active)* | Adjust volume — wider gap = louder |

The session is **off by default**. Nothing is dispatched to Spotify until you activate it with the open-palm hold. This prevents accidental triggers while typing, eating, or gesturing in conversation.

---

## Key design decisions

**Session toggle, not per-gesture activation.** Hold an open palm for 1.5 seconds once to activate the session; all subsequent gestures fire instantly until you hold the palm again to deactivate. This eliminates the "Midas Touch" effect.

**Velocity-based swipe detection.** Swipes are detected by computing the signed horizontal velocity of the wrist landmark over a 5-frame rolling window, not by a heavy classifier. This makes swipe detection fast and tunable without retraining.

**Rearm cooldown.** After a swipe fires, the detector locks for 800 ms so the hand returning to centre does not trigger a reverse skip.

**Spotify Premium + OS media key fallback.** If the Spotify Web API returns a 403 (no Premium) or 429 (rate limited), Gestureify automatically falls back to simulating OS media keys, so it works with the Spotify desktop client regardless of account tier.

**JAX-vectorised geometry.** The two hot-path functions (`fingertip_wrist_ratio`, `pinch_distance`) are compiled with `@jax.jit` and operate on stacked landmark arrays via `jnp.linalg.norm` — zero Python loops in the 30 FPS critical path.

---

## Requirements

- Python **3.10+** (required for `slots=True` dataclasses and `str | None` union syntax)
- [uv](https://docs.astral.sh/uv/) package manager
- A webcam
- A **Spotify Premium** account (the Spotify Web API requires Premium for playback control; Free accounts use the media-key fallback automatically)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/LeoMattosMartins/cantrips-2.git
cd cantrips-2
```

### 2. Install dependencies with uv

```bash
uv sync
```

This reads `pyproject.toml`, creates a virtual environment in `.venv/`, and installs all dependencies including MediaPipe, JAX (CPU), OpenCV, and Pydantic.

To also install development dependencies (pytest, ruff, mypy):

```bash
uv sync --extra dev
```

### 3. Install the Monocraft font (optional but recommended)

The HUD uses the [Monocraft](https://github.com/IdreesInc/Monocraft) pixel-style monospace font, which is bundled in `gestureify/assets/fonts/`. On Linux, the font is registered automatically at startup via fontconfig. On macOS or Windows, install it manually by double-clicking `gestureify/assets/fonts/Monocraft.ttf` and clicking **Install**.

---

## Spotify setup

Gestureify uses the [Spotify Web API](https://developer.spotify.com/documentation/web-api) with the **PKCE Authorization Code** flow — no client secret is ever stored or transmitted.

### Step 1 — Create a Spotify app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create app**.
3. Fill in any name and description (e.g. *Gestureify*).
4. Under **Redirect URIs**, add exactly: `http://localhost:8080/callback`
5. Under **APIs used**, select **Web API**.
6. Click **Save**.

### Step 2 — Copy your Client ID

On your app's dashboard page, copy the **Client ID** (a 32-character hex string). You do **not** need the Client Secret.

### Step 3 — Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and set your Client ID:

```dotenv
SPOTIFY_CLIENT_ID=your_32_character_client_id_here
```

All other values have sensible defaults. The full list of available overrides is documented in `.env.example`.

### Step 4 — First launch and authentication

```bash
uv run gestureify
```

On first launch, Gestureify will open your browser to the Spotify authorisation page. After you approve access, the browser redirects to `localhost:8080/callback` and the token is cached at `~/.gestureify/tokens.json` (permissions `0600`). Subsequent launches reuse the cached token and refresh it silently — you will not need to log in again unless you revoke access.

---

## Running the app

```bash
uv run gestureify
```

Or, with the virtual environment activated manually:

```bash
source .venv/bin/activate
python main.py
```

### Environment variables

All settings can be overridden via environment variables or the `.env` file. Pydantic validates every value at startup and prints a structured error if anything is wrong.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SPOTIFY_CLIENT_ID` | *(required)* | Your Spotify app Client ID |
| `SPOTIFY_REDIRECT_URI` | `http://localhost:8080/callback` | Must match the Dashboard exactly |
| `CAMERA_INDEX` | `0` | OpenCV camera index (try `1` or `2` for external webcams) |
| `LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

---

## Running the tests

```bash
uv run pytest tests/ -v
```

All 52 tests run without a camera, Spotify credentials, or a display. The test suite covers JAX geometry utilities, the gesture classifier (including the `ClassifyResult` API and thumb-exclusion behaviour), session gate FSM, swipe detector FSM, and timing utilities.

---

## Tuning

All thresholds live in `gestureify/config/settings.py` as `Final[T]` constants. Override any of them at runtime via `.env`. Key values:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `WAKE_HOLD_SECONDS` | `1.5` | Seconds to hold open palm to toggle session |
| `SWIPE_VELOCITY_THRESHOLD` | `0.04` | Minimum wrist velocity to register a swipe |
| `SWIPE_COOLDOWN_SECONDS` | `0.8` | Rearm window after a swipe fires |
| `PINCH_DISTANCE_THRESHOLD` | `0.12` | Normalised gap to enter pinch mode |
| `VOLUME_API_THROTTLE_SECONDS` | `0.3` | Minimum interval between volume API calls |

---

## Architecture

```
main.py
├── AppConfig (Pydantic v2)        — validated runtime config
├── SpotifyAuth (PKCE OAuth)       — token acquisition & refresh
├── CVPipeline (daemon thread)
│   ├── CameraCapture              — OpenCV camera wrapper
│   ├── LandmarkExtractor          — MediaPipe Hands (isolated)
│   ├── GestureClassifier          — JAX-vectorised static gestures
│   ├── SessionGate                — open-palm toggle FSM
│   └── SwipeDetector              — velocity-window swipe FSM
├── PlaybackController             — Spotify API + media-key fallback
└── HUDOverlay (main thread)       — Tkinter / Monocraft terminal overlay
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for state machine diagrams and threading model details.

### Threading model

```
Main thread  ──► Tkinter HUD event loop
                      ▲
                      │  queue.Queue[HUDMessage]  (non-blocking put)
                      │
CV thread    ──► Camera capture → MediaPipe → Gesture FSM → PlaybackController
```

The CV thread is a **daemon thread** — it is killed automatically if the main thread exits. The `stop_event` threading flag provides a clean cooperative shutdown path.

---

## Project structure

```
cantrips-2/
├── main.py                         # Entry point
├── pyproject.toml                  # uv / hatchling build config + ruff/mypy
├── .env.example                    # Environment variable template
├── gestureify/
│   ├── assets/
│   │   ├── font_loader.py          # Registers Monocraft with OS font system
│   │   └── fonts/
│   │       ├── Monocraft.ttf       # Monocraft regular (bundled, OFL-1.1)
│   │       └── Monocraft-Bold.ttf  # Monocraft bold (bundled, OFL-1.1)
│   ├── auth/
│   │   ├── pkce.py                 # RFC 7636 PKCE helpers (pure functions)
│   │   ├── token_store.py          # Atomic token cache (chmod 0600)
│   │   └── spotify_auth.py         # PKCE OAuth flow + silent refresh
│   ├── config/
│   │   ├── settings.py             # Compile-time constants (Final[T])
│   │   ├── env_loader.py           # .env parsing with validation
│   │   └── app_config.py           # Pydantic v2 runtime config model
│   ├── controller/
│   │   ├── spotify_client.py       # Typed Spotify Web API wrapper
│   │   ├── media_keys.py           # OS media-key fallback (pynput)
│   │   └── playback_controller.py  # Command dispatcher + throttling
│   ├── cv_engine/
│   │   ├── capture.py              # OpenCV camera wrapper
│   │   ├── landmark_extractor.py   # MediaPipe isolation layer
│   │   ├── gesture_classifier.py   # JAX-backed static gesture classifier
│   │   ├── session_gate.py         # Open-palm session toggle FSM
│   │   ├── swipe_detector.py       # Velocity-window swipe FSM
│   │   └── pipeline.py             # Per-frame CV orchestrator
│   ├── hud/
│   │   └── overlay.py              # Tkinter terminal-style HUD (Monocraft)
│   └── utils/
│       ├── geometry.py             # JAX-vectorised landmark math
│       ├── timing.py               # RateGate, Stopwatch
│       └── logger.py               # Logging configuration
├── tests/
│   ├── test_geometry.py
│   ├── test_classifier.py
│   └── test_timing_and_state.py
└── docs/
    └── ARCHITECTURE.md
```

---

## Coding standards

This codebase follows the spirit of the **NASA Power of Ten Rules**:

1. Simple control flow — no `goto`, no deep recursion.
2. All loops and data structures are bounded (fixed-size ring buffers, bounded callback servers).
3. No dynamic memory allocation after initialisation in the hot path (`slots=True` dataclasses, pre-allocated JAX arrays).
4. Functions are kept short (≤ 30 lines of logic) with a single clear responsibility.
5. Data scope is minimised — no global mutable state outside the standard `logging` module.
6. Return values and error conditions are always checked.
7. Typed exceptions (`SpotifyPremiumRequired`, `SpotifyRateLimited`) replace silent failures.
8. All public functions and classes have docstrings with parameter and return type documentation.

---

## Troubleshooting

**"No active device found" / volume commands silently fail**
Spotify must be open and playing on a device. Open Spotify on your computer or phone and start playing something before launching Gestureify.

**"Spotify Premium required"**
The Spotify Web API's playback control endpoints require a Premium subscription. On Free accounts, Gestureify automatically falls back to OS media keys for play/pause and skip, but volume control is unavailable.

**The camera is not detected**
Try setting `CAMERA_INDEX=1` (or `2`) in your `.env` file. On Linux, list available devices with `v4l2-ctl --list-devices`.

**The HUD font looks wrong**
Install the Monocraft font manually: double-click `gestureify/assets/fonts/Monocraft.ttf` and install it system-wide, then restart Gestureify.

**Gestures are triggering when I don't want them to**
Make sure the session is OFF (the HUD should show `SESSION OFF` in grey). Hold an open palm for ~1.5 seconds to deactivate.

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for the full text.

The bundled Monocraft font (`gestureify/assets/fonts/`) is licensed under the [SIL Open Font License 1.1](https://openfontlicense.org/) by [@IdreesInc](https://github.com/IdreesInc).

---

## Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — hand landmark detection
- [JAX](https://github.com/google/jax) — vectorised numerical computation
- [Pydantic](https://docs.pydantic.dev/) — runtime configuration validation
- [Monocraft](https://github.com/IdreesInc/Monocraft) by [@IdreesInc](https://github.com/IdreesInc) — the pixel-style monospace font used in the HUD
- [Spotify Web API](https://developer.spotify.com/documentation/web-api) — playback control
