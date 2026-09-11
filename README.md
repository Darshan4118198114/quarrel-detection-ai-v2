# Quarrel Detection AI - Advanced Multi-Modal Surveillance System

An intelligent, real-time computer vision and audio surveillance system designed to detect physical aggression, confrontations, yelling, and quarrels between individuals in video feeds.

---

## 🚀 Key Features & Accuracy Enhancements

1. **Scale-Invariant Facial Expression & Shouting Analysis**:
   - Computes normalized **Mouth Aspect Ratio (MAR)** to identify yelling, shouting, and verbal aggression regardless of how close or far the person is from the camera.
   - Computes **Eyebrow Furrow Ratio** and brow-to-eye compression to detect anger and scowling expressions.
   - De-biases facial motion by subtracting head translation vectors, preventing regular head turns from falsely triggering alerts.

2. **Persistent Multi-Person Centroid Tracking**:
   - Prevents identity swapping between subjects across frames.
   - Tracks individual motion history, velocity, and bounding boxes.

3. **Spatial Proximity & Confrontation Corridor Dynamics**:
   - Analyzes true Euclidean proximity between individuals.
   - Measures dynamic motion energy in the interaction corridor (space between subjects) where shoving, striking, or arm waving occurs.
   - Detects rapid convergence/lunging velocity when subjects encroach into personal space.

4. **Multi-Modal Audio Shouting Detection (Optional)**:
   - Integrates real-time ambient noise tracking via microphone (`sounddevice`).
   - Flags sudden shouting spikes that coincide with physical confrontation, significantly reducing false positives.
   - Fails gracefully if no microphone is available.

5. **Temporal Exponential Moving Average (EMA) Risk Engine**:
   - Smooths frame-by-frame fluctuations to eliminate flickering false alarms.
   - Multi-tier state machine: `NORMAL` $\to$ `ATTENTION` $\to$ `CONFRONTATION` $\to$ `QUARREL DETECTED`.

6. **Production Alert System & Surveillance HUD**:
   - Sleek tactical surveillance overlay showing live risk gauge (0–100%), proximity lines, and subject IDs.
   - Non-blocking audio alert chime on confirmed quarrels.
   - Automatic timestamped incident snapshots saved to `incidents/quarrel_YYYYMMDD_HHMMSS.jpg`.
   - Structured audit logging to `events.log`.

---

## 📦 Installation

### 1. Clone or Extract Project
```bash
cd quarrel-detection-ai
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage & Running

### Option 1: Default Laptop / USB Webcam
```bash
python main.py
```
Or specify camera index:
```bash
python main.py --source 0
```

### Option 2: Mobile Phone Camera via Wi-Fi

#### Using IP Webcam (Android - Recommended):
1. Install **IP Webcam** from Google Play Store on your phone.
2. Connect both phone and PC to the same Wi-Fi network.
3. Open the app, scroll to the bottom, and tap **Start server**.
4. Note the IP address displayed on your phone screen (e.g. `http://192.168.1.15:8080`).
5. Run:
```bash
python main.py --source http://192.168.1.15:8080/video
```

#### Using DroidCam (Android / iOS):
1. Install and open **DroidCam** on your phone.
2. Run:
```bash
python main.py --source http://<PHONE_IP>:4747/video
```

### Option 3: Analyze a Video File
```bash
python main.py --source path/to/surveillance_video.mp4
```

### Option 4: Record Processed Output Video with HUD
```bash
python main.py --source 0 --record output_surveillance.mp4
```

### Option 5: Synthetic Offline Test Mode (No Camera Required)
```bash
python main.py --source test
```

---

## ⚙️ Configuration Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--source`, `-s` | Video source: index (`0`), URL (`http://...`), file (`.mp4`), or `test` | `0` |
| `--sensitivity` | Risk score multiplier (`0.8` for fewer alerts, `1.2` for higher sensitivity) | `1.0` |
| `--no-audio` | Disable microphone shouting detection | False |
| `--headless` | Run in background without GUI window (for servers / benchmarks) | False |
| `--record` | Path to save recorded video with HUD overlay | None |
| `--max-frames` | Automatically stop after processing N frames | None |

---

## ⌨️ Hotkeys during Live Surveillance

- `q` or `ESC`: Exit application cleanly.
- `s`: Capture manual snapshot of current frame to `incidents/`.

---

## 🧪 Running Automated Tests

A complete automated unit and integration test suite is included:
```bash
python test_detector.py
```
This validates face tracking identity persistence, scale invariance, motion calculation, and alert creation.

---

## 📁 Project Structure

```
quarrel-detection-ai/
│
├── main.py                     # Main application entry point & surveillance pipeline
├── requirements.txt            # Dependency specifications
├── README.md                   # Project documentation
├── test_detector.py            # Automated test suite
├── create_zip.py               # Clean package bundler
│
├── detection/
│   ├── face_tracker.py         # Multi-person centroid tracker with persistent IDs
│   ├── expression_detector.py  # Scale-invariant MAR, brow furrow, and head motion
│   └── quarrel_detector.py     # Multi-modal fusion risk engine & state machine
│
├── motion/
│   └── motion_detector.py      # Corridor motion energy & approach velocity
│
├── audio/
│   └── audio_detector.py       # Real-time microphone shouting & decibel detector
│
├── alerts/
│   └── alert_system.py         # Chime alert, incident snapshot capture, and cooldown
│
├── utils/
│   ├── visualizer.py           # Modern surveillance HUD overlay renderer
│   ├── logger.py               # Structured audit event logging (events.log)
│   └── helpers.py              # Euclidean & math helpers
│
└── incidents/                  # Auto-saved snapshots on confirmed quarrels
```
