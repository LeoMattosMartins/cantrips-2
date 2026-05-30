# Gestureify 🎵🤚

> Control Spotify with hand gestures — no mouse, no keyboard, no alt-tab.

Gestureify is a local desktop utility that uses your webcam and Google MediaPipe to translate hand gestures into Spotify playback commands in real time. It runs entirely on your machine (no cloud CV calls) and communicates with Spotify via the Web API using a secure PKCE OAuth2 flow.

---

## Features

| Gesture | Action |
| :--- | :--- |
| **Open palm held for 1.5 s** | Toggle gesture session ON / OFF |
| **Closed fist** | Pause playback |
| **Swipe right** (fast left→right) | Skip to next track |
| **Swipe left** (fast right→left) | Go to previous track |
| **Pinch** (thumb + index) | Adjust volume (gap = level) |

### Key design decisions

- **Session toggle, not per-gesture activation.** Hold an open palm for 1.5 seconds once to activate the session; all subsequent gestures fire instantly until you hold the palm again to deactivate. This eliminates the "Midas Touch" effect — accidental triggers from everyday hand movements.
- **Velocity-based swipe detection.** Swipes are detected by computing the signed horizontal velocity of the wrist landmark over a 5-frame rolling window, not by a heavy classifier. This makes swipe detection fast and tunable without retraining.
- **Rearm cooldown.** After a swipe fires, the detector locks for 800 ms so the hand returning to centre does not trigger a reverse skip.
- **Spotify Premium + OS media key fallback.** If the Spotify Web API returns a 403 (no Premium) or 429 (rate limited), Gestureify automatically falls back to simulating OS media keys, so it works with the Spotify desktop client regardless of account tier.

---

## Requirements

- Python **3.10 or 3.11** (MediaPipe constraint)
- A webcam
- A Spotify account (Premium required for Web API volume control; Free accounts use the media-key fallback)
- A Spotify Developer App (free to create — see Setup)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/LeoMattosMartins/cantrips-2.git
cd cantrips-2
```

### 2. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a Spotify Developer App

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create App**.
3. Set **App name** to anything (e.g. `Gestureify`).
4. Under **Redirect URIs**, add exactly: `http://localhost:8080/callback`
5. Save. Copy your **Client ID**.

### 5. Configure your environment

```bash
cp .env.example .env
```

Open `.env` and set your Client ID:

```
SPOTIFY_CLIENT_ID=your_client_id_here
```

### 6. Run

```bash
python main.py
```

On first launch, your browser will open to the Spotify login page. After you authorise the app, tokens are cached locally in `.spotify_token_cache.json` — you will not need to log in again unless you revoke access.

---

## Usage

1. Start the app. The floating HUD appears in the top-left corner of your screen.
2. The status ring shows **IDLE** (grey) — gestures are not yet active.
3. Hold an **open palm** in front of your webcam for ~1.5 seconds. The ring turns **green** and the label flashes "Session ON".
4. Use gestures freely. The HUD shows the hand skeleton and flashes action labels.
5. Hold an open palm again for ~1.5 seconds to deactivate. The ring returns to grey.

### Tuning

All thresholds live in `gestureify/config/settings.py`. No magic numbers are scattered elsewhere. Key values to adjust:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `WAKE_HOLD_SECONDS` | `1.5` | Seconds to hold open palm to toggle session |
| `SWIPE_VELOCITY_THRESHOLD` | `0.04` | Minimum velocity to register a swipe |
| `SWIPE_COOLDOWN_SECONDS` | `0.8` | Rearm window after a swipe fires |
| `PINCH_DISTANCE_THRESHOLD` | `0.12` | Normalised gap to enter pinch mode |
| `VOLUME_API_THROTTLE_SECONDS` | `0.3` | Minimum interval between volume API calls |

---

## Architecture

```
cantrips-2/
├── main.py                         # Entry point; wires all modules; threading model
├── requirements.txt
├── .env.example
└── gestureify/
    ├── config/
    │   ├── settings.py             # All constants and thresholds (single source of truth)
    │   └── env_loader.py           # .env parsing and required-key validation
    ├── auth/
    │   ├── pkce.py                 # RFC 7636 PKCE verifier/challenge generation
    │   ├── token_store.py          # Atomic JSON token cache (mode 0o600)
    │   └── spotify_auth.py         # Full PKCE OAuth flow + token refresh
    ├── cv_engine/
    │   ├── capture.py              # OpenCV VideoCapture wrapper (context manager)
    │   ├── landmark_extractor.py   # MediaPipe Hands wrapper → plain (x,y) tuples
    │   ├── gesture_classifier.py   # Static gesture classification (fist/palm/pinch)
    │   ├── session_gate.py         # Open-palm session toggle state machine
    │   ├── swipe_detector.py       # Velocity-window swipe detection + rearm FSM
    │   └── pipeline.py             # Orchestrates all CV components per frame
    ├── controller/
    │   ├── spotify_client.py       # Typed Spotify Web API wrapper (urllib only)
    │   ├── media_keys.py           # OS media key fallback via pynput
    │   └── playback_controller.py  # Dispatcher: API → fallback, throttling, cooldowns
    ├── hud/
    │   └── overlay.py              # Tkinter floating overlay (thread-safe queue model)
    └── utils/
        ├── logger.py               # Root logger configuration
        ├── geometry.py             # Pure geometry helpers for landmark arithmetic
        └── timing.py               # RateGate and Stopwatch primitives
```

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

## Running Tests

```bash
pytest tests/ -v
```

Tests cover geometry helpers, timing primitives, the session gate state machine, and the swipe detector — all without requiring a camera or Spotify credentials.

---

## Coding Standards

This codebase follows the spirit of the **NASA Power of Ten Rules**:

1. Simple control flow — no `goto`, no deep recursion.
2. All loops and data structures are bounded (fixed-size ring buffers, bounded callback servers).
3. No dynamic memory allocation after initialisation in the hot path.
4. Functions are kept short (≤ 30 lines of logic) with a single clear responsibility.
5. Data scope is minimised — no global mutable state outside the standard `logging` module.
6. Return values and error conditions are always checked.
7. Typed exceptions (`SpotifyPremiumRequired`, `SpotifyRateLimited`) replace silent failures.
8. All public functions and classes have docstrings with parameter and return type documentation.

---

## License

MIT — see `LICENSE` for details.
