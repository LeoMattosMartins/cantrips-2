# Gestureify — Architecture Deep Dive

This document describes the internal design decisions, data flows, and state machines in detail.

---

## 1. Module Dependency Graph

```
main.py
 ├── gestureify.config          (no internal deps)
 ├── gestureify.utils           (no internal deps)
 ├── gestureify.auth
 │    └── gestureify.config
 │    └── gestureify.utils
 ├── gestureify.cv_engine
 │    ├── gestureify.config
 │    └── gestureify.utils
 ├── gestureify.controller
 │    ├── gestureify.auth
 │    ├── gestureify.config
 │    └── gestureify.utils
 └── gestureify.hud
      ├── gestureify.config
      └── gestureify.cv_engine.session_gate  (SessionState enum only)
```

Dependencies are strictly **downward** — no circular imports.

---

## 2. Session Gate State Machine

```
                ┌─────────────────────────────────┐
                │             IDLE                │
                │  (no commands dispatched)       │
                │                                 │
                │  On each frame:                 │
                │    if gesture == OPEN_PALM:      │
                │      start/advance hold timer   │
                │    else:                        │
                │      reset hold timer           │
                └──────────────┬──────────────────┘
                               │
                               │  hold_timer >= WAKE_HOLD_SECONDS
                               │  (timer resets after transition)
                               ▼
                ┌─────────────────────────────────┐
                │            ACTIVE               │
                │  (commands dispatched)          │
                │                                 │
                │  On each frame:                 │
                │    if gesture == OPEN_PALM:      │
                │      start/advance hold timer   │
                │    else:                        │
                │      reset hold timer           │
                └──────────────┬──────────────────┘
                               │
                               │  hold_timer >= WAKE_HOLD_SECONDS
                               ▼
                             IDLE  (cycle repeats)
```

The open-palm gesture is **consumed by the gate** and never forwarded as a playback command, preventing the wake gesture from accidentally resuming a paused track.

---

## 3. Swipe Detector State Machine

```
              ┌────────────────────────────────────┐
              │             READY                  │
              │                                    │
              │  Buffer: deque(maxlen=5)            │
              │  Each frame: append wrist_x        │
              │  If |velocity| > threshold:        │
              │    emit SwipeDirection              │
              │    → enter COOLDOWN                │
              │  If wrist_x is None:               │
              │    clear buffer (stay READY)       │
              └──────────────┬─────────────────────┘
                             │  swipe detected
                             ▼
              ┌────────────────────────────────────┐
              │           COOLDOWN                 │
              │                                    │
              │  Buffer cleared immediately.       │
              │  All updates return None.          │
              │  After SWIPE_COOLDOWN_SECONDS:     │
              │    → return to READY               │
              └────────────────────────────────────┘
```

**Why clear the buffer on cooldown entry?**
Without clearing, the samples accumulated during the forward swipe would still be in the buffer when the cooldown ends. The first new sample appended would compute velocity against those stale values, potentially triggering a phantom swipe in the opposite direction.

---

## 4. Threading Model

```
┌──────────────────────────────────────────────────────────────┐
│  Main Thread (Tkinter)                                       │
│                                                              │
│  hud.run()  →  root.mainloop()                               │
│                    │                                         │
│                    │  root.after(16ms, _poll)                │
│                    ▼                                         │
│              _poll() drains queue.Queue[HUDMessage]          │
│              and calls _render(msg) synchronously            │
└──────────────────────────────────────────────────────────────┘
                         ▲
                         │  hud_queue.put_nowait(msg)
                         │  (drops frame if queue full — never blocks)
                         │
┌──────────────────────────────────────────────────────────────┐
│  CV Thread (daemon)                                          │
│                                                              │
│  for frame in capture.frames():                              │
│      result = pipeline.process(frame)                        │
│      dispatch commands to PlaybackController                 │
│      enqueue HUDMessage                                      │
└──────────────────────────────────────────────────────────────┘
```

**Why a daemon thread?**
If the user closes the HUD window, the main thread exits. A daemon thread is automatically killed at that point, ensuring the camera is released via `capture.close()` in the `finally` block of `main()`.

---

## 5. Volume Throttling

The pinch gesture produces a continuous stream of `pinch_gap` values at 30 FPS. Without throttling, this would fire ~30 Spotify API calls per second, immediately triggering a 429 rate-limit error.

The `RateGate(min_interval=0.3)` in `PlaybackController` allows at most ~3 volume API calls per second. The HUD volume bar updates every frame (local only) so the visual feedback remains smooth even when API calls are throttled.

---

## 6. Token Security

- Tokens are stored in `.spotify_token_cache.json` with Unix permissions `0o600` (owner read/write only).
- The file is written atomically: a temporary file is written first, then renamed into place. This prevents a partial write from corrupting the cache.
- The `.env` file (containing the Client ID) is listed in `.gitignore` and must never be committed.
- No client secret is ever used; the PKCE flow is secret-free by design.
