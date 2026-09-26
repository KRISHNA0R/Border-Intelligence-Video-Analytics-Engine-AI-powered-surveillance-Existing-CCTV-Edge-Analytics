<p align="center">
  <img src="assets/logo.png" width="250" alt="BorderEye logo" />
</p>

<h1 align="center">🛡️ BorderEye — Border Intelligence &amp; Video Analytics Engine</h1>

<p align="center">
  <b>The border never sleeps — and neither do we.</b><br/>
  <i>सतर्क सीमा, सुरक्षित देश।</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.14" />
  <img src="https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&amp;logoColor=white" alt="Streamlit 1.58" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FFC107?logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&amp;logoColor=white" alt="FastAPI REST + WebSocket" />
  <img src="https://img.shields.io/badge/tests-167%20passing-4C1?logo=pytest&amp;logoColor=white" alt="167 tests passing" />
  <img src="https://img.shields.io/badge/Smart%20India%20Hackathon-2026-0D3B66" alt="Smart India Hackathon 2026" />
</p>

<p align="center">
  <b>Smart India Hackathon 2026</b> · AI-powered surveillance for border checkpoints and border roads<br/>
  Edge-first · Bandwidth-honest · Tamper-evident · CPU-only
</p>

> **Smart India Hackathon 2026 · Problem Statement SIH26187**
> *"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three."*

---

## 📌 What It Does

BorderEye is a **software layer over existing CCTV cameras** at border outposts. It watches every frame with AI, raises alerts the instant something happens, and proves every alert is genuine with a tamper-evident ledger. **No new cameras. No cloud dependency. No GPU needed.**

All inference runs at the edge. Only compact event metadata (~200 bytes) travels upstream — **video never leaves the post**.

---

## ✅ What Actually Works

| Feature | Status | Module |
|---|---|---|
| YOLOv8 object detection (persons, vehicles, autorickshaw) | ✅ Working | `src/edge/detector.py` |
| ByteTrack multi-object tracking with persistent IDs | ✅ Working | `src/edge/tracker.py` |
| Virtual fences + pen-drawn tripwire designer | ✅ Working | `src/edge/fence.py` · `web_demo.py` |
| ANPR — plate proposals + EasyOCR consensus | ✅ Working | `src/edge/anpr.py` |
| Face detection (Haar cascade, honest accuracy label) | ✅ Working | `modules/face_intel.py` |
| Night mode — CLAHE low-light enhance (auto 19:00–06:00) | ✅ Working | `src/edge/pipeline.py` |
| Camera signal-loss / tamper detection | ✅ Working | `src/edge/signal.py` |
| SHA-256 tamper-evident hash chain + self-chained anchor | ✅ Working | `src/edge/hashchain.py` |
| Offline outbox queue + auto-sync when link returns | ✅ Working | `web_demo.py` |
| Threat score 0–100 with documented risk formula | ✅ Working | `web_demo.py` |
| Groq AI alert summaries (English + Hindi action line) | ✅ Working | `web_demo.py` |
| One-click PDF incident reports (works offline) | ✅ Working | `web_demo.py` |
| Behavioral analytics — class, hourly, freshness charts | ✅ Working | `web_demo.py` |
| Edge map with real latitude/longitude sites | ✅ Working | `modules/border_map.py` |
| Audio threat cues (browser beep, no assets needed) | ✅ Working | `modules/audio_threat.py` |
| Camera health + system checks | ✅ Working | `modules/checks.py` |
| FastAPI REST + WebSocket API | ✅ Working | `src/backend/api.py` |
| Streamlit command dashboard (login-gated) | ✅ Working | `web_demo.py` |
| Docker + Streamlit Cloud deployment | ✅ Working | `Dockerfile` · `docs/DEPLOY.md` |
| Test suite — pipeline, API, converters | ✅ 167 passing | `tests/` |

Roadmap items that are documented but not yet built live in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## 🆚 Why BorderEye Is Different

| Existing CCTV / AI platforms | 🛡️ BorderEye |
|---|---|
| Assume good bandwidth, stream video 24×7 | **Bandwidth-honest** — only ~200 bytes of metadata travel; video never leaves the edge |
| Go blind silently when the network drops | **Offline-first** — local queue + auto-sync; the system never goes silent |
| Alerts you must take on faith | **Tamper-evident** — every alert hash-chained; one click verifies the whole log |
| Fixed rectangular zones | **Pen-tool tripwires** — draw any free-form line on the live frame |
| Generic COCO classes | **India-tuned** — autorickshaw class, Hindi action lines, IDD fine-tuning |
| GPU servers, lakh-rupee setups | **CPU-only, tiered hardware** — full post under ~$250, remote node under ~$50 |
| Black-box "AI threat score" | **Documented risk formula** — every point explained, judges can audit it |
| Demo footage passed off as live | **Honest labelling** — every non-live feed carries a visible `SIMULATED` tag |

---

## 🗺️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 · EDGE NODE  (checkpost — all inference runs here)          │
│                                                                     │
│    Camera ──▶ YOLOv8 ──▶ ByteTrack ──▶ Fence · Tripwire · ANPR      │
│                              │          Faces · Night · Signal-loss │
│                              ▼                                      │
│                 SHA-256 hash-chained ledger                         │
│                 └─ link DOWN: events queue to data/outbox.jsonl     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  ~200 bytes of metadata only
                               │  (video never leaves the edge)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2 · BACKEND                                                   │
│                                                                     │
│    FastAPI  ──▶ /api/alerts · /api/tracks · /api/chain · /api/…     │
│    WebSocket ──▶ /ws/live   (live detection stream)                 │
│    link UP ──▶ outbox drains, alerts flip to ✅ synced              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3 · COMMAND DASHBOARD  (Streamlit)                            │
│                                                                     │
│    Live feed + HUD ── Alert console ── Analytics ── Edge map        │
│    Tripwire designer ── PDF report ── Camera health ── SOS          │
│    AI summaries ── Edge ledger verify ── Sim controls ── Logout     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow (per frame):**

```
frame → night-enhance → detect → track → fence/tripwire → ANPR/faces → hash-chain
      → annotated frame + alert → link UP: sync   |   link DOWN: queue → auto-sync
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/ROADMAP.md`](docs/ROADMAP.md) · SIH brief: [`docs/BorderEye_SIH_Brief.pdf`](docs/BorderEye_SIH_Brief.pdf)

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Dashboard (default)
python main.py dashboard            # → http://localhost:8501

# 3. API server
python main.py server               # → http://localhost:8000  (/docs = Swagger)

# 4. Edge demo on a video
python main.py demo --video path/to/video.mp4

# 5. Tests (no model weights downloaded)
python -m pytest                    # 167 tests, ~4 s

# 6. Docker
docker-compose up dashboard         # dashboard only
docker-compose up                   # full stack (dashboard + API)

# 7. Dev shortcuts
make help                           # every command, one list
```

**Login** — demo gate: any email + any 6-digit password (e.g. `admin@gmail.com` / `123456`).

**Groq key (optional)** — put it in `.streamlit/secrets.toml` (never commit it):

```toml
GROQ_API_KEY = "gsk_..."
```

**Deploy** — free-tier hosting steps in [`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## 🖥️ Inside the Dashboard

**0. 🔐 Login gate** — full app locked behind the demo login; logout wipes the session.

**1. Header** — logo + title, ➕ Add Camera/Footage (RTSP with name/location, or video upload), link/SOS banners, scrolling status ticker.

**2. 📹 Live Detection Feed (4 sources)** — 🎬 Demo Mode · 🎥 Upload Video (4 tabs, play + progress + single-frame OCR inspect) · 📷 Webcam · 🌐 RTSP Stream. Every frame: night-enhance → YOLO + ByteTrack → fence + tripwire checks → face boxes → HUD overlay.

**3. 🚨 Alert Console** — severity filter pills, severity beep (with Test Beep), ACK / ACK-ALL workflow, threat score 0–100 (expandable "why"), timestamp + event ID, ✅ synced / 📤 queued status.

**4. 📊 Dashboard** — metrics, full alert JSON, low-bandwidth payload preview (exact bytes on the wire), AI summary, hash-chain verify, 24-hour activity chart.

**5. 📈 Behavioral Analytics** — frames/persons/vehicles/faces, class + hourly charts.

**6. Sidebar** — model status · night/face toggles · simulate buttons · link toggle + outbox + sync · SOS + webhook · footage/RTSP libraries · edge ledger + anchor · PDF report · AI summariser · camera health · theme toggle · logout.

**7. ✏️ Tripwire Designer** — sketch any line on a 640×480 canvas → save → magenta lines on the live feed → crossings fire tracked alerts.

**8. ⌨️ Operator shortcuts** — `1`–`8` switch panels instantly; dark-mode toggle; camera-wall freshness tags.

**9. 📡 Offline flow** — link OFF: AI runs, events queue in `data/outbox.jsonl` → link ON: auto-sync, cards flip to ✅.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System status |
| GET | `/api/alerts` | Recent alerts |
| GET | `/api/alerts/verify` | Verify the SHA-256 hash chain |
| GET | `/api/cameras` | Camera status |
| GET | `/api/signal/thresholds` | Signal-loss thresholds |
| POST | `/api/signal/thresholds` | Update thresholds |
| GET | `/api/fences` | Virtual fence zones |
| POST | `/api/fences` | Create a fence zone |
| DELETE | `/api/fences/{zone_name}` | Remove a fence zone |
| GET | `/api/tracks` | Active tracked objects |
| GET | `/api/plates` | ANPR results |
| GET | `/api/chain` | Full hash chain |
| GET | `/api/chain/export` | Export chain as JSON |
| POST | `/api/process/video` | Process uploaded video |
| WS | `/ws/live` | Live detection stream |

---

## 📁 Project Structure

```
BorderEye/
├── main.py                    # Entry point (dashboard | server | demo)
├── web_demo.py                # Streamlit command dashboard
├── requirements.txt           # Core deps (CPU-only torch wheel)
├── requirements-ml.txt        # GPU training extras
├── Makefile                   # make help — every dev command
├── Dockerfile                 # Container build
├── docker-compose.yml         # Dashboard + API stack
├── packages.txt · start.sh    # Streamlit Cloud deployment
│
├── src/
│   ├── config.py              # System configuration (dataclasses)
│   ├── edge/                  # Edge inference modules
│   │   ├── detector.py        # YOLOv8 detection (custom classes)
│   │   ├── tracker.py         # ByteTrack persistent IDs
│   │   ├── fence.py           # Virtual fence zones
│   │   ├── anpr.py            # ANPR with OCR consensus
│   │   ├── signal.py          # Signal-loss / tamper detection
│   │   ├── hashchain.py       # SHA-256 tamper-evident log
│   │   └── pipeline.py        # Frame pipeline + night mode
│   ├── backend/
│   │   └── api.py             # FastAPI REST + WebSocket
│   └── utils/
│       └── logger.py          # Structured logging
│
├── modules/                   # Dashboard intelligence modules
│   ├── face_intel.py          # Face detection panel
│   ├── audio_threat.py        # Threat beep (generated WAV, no assets)
│   ├── border_map.py          # Edge map with real lat/lng sites
│   └── checks.py              # Camera health + system checks
│
├── scripts/                   # Dataset prep, training, export
│   ├── idd_to_yolo.py         # IDD-Detection → YOLO
│   ├── exdark_to_yolo.py      # ExDark → low-light YOLO
│   ├── plates_to_yolo.py      # Plate datasets → single class
│   ├── anpr_augmentation.py   # Plate augmentation pipeline
│   ├── train.py               # YOLOv8 fine-tuning
│   ├── evaluate.py            # mAP day/night, ANPR accuracy
│   └── export_onnx.py         # ONNX export for edge deployment
│
├── tests/                     # 167 tests — no model downloads
├── docs/                      # ARCHITECTURE · ROADMAP · DEPLOY · SIH brief (PDF)
├── pitch/                     # Deck, demo script, recording guide, timeline
├── assets/logo.png            # BorderEye logo
├── models/                    # detection.pt / plate.pt weights
├── data/                      # datasets · outbox · ledger
└── .streamlit/ · config/ · logs/
```

---

## 🧠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| 🖥️ Dashboard | **Streamlit 1.58** | Command-centre UI — feed, alerts, analytics, tripwire canvas |
| ✏️ Drawing | **streamlit-image-coordinates** | Click-to-draw tripwire designer (1:1 with detection frame) |
| 🔍 Detection | **YOLOv8 (Ultralytics 8.4)** | Person + vehicle detection, `detection.pt` fine-tuned on IDD (7 classes) |
| 🎯 Tracking | **ByteTrack (built-in)** | Persistent IDs (`PERSON #3`) via `model.track(persist=True)` |
| 🔢 ANPR | **plate.pt + EasyOCR 1.7** | Plate-region proposals + OCR text |
| 🧑 Faces | **OpenCV Haar cascade 4.14** | Face boxes (classical ML — honest accuracy label in UI) |
| 🌙 Night | **CLAHE (OpenCV)** | Low-light enhance, auto-active 19:00–06:00 |
| 🧠 LLM | **Groq API (`gpt-oss-20b`)** | Alert → 2-line English summary + Hindi `कार्रवाई` line |
| ⛓️ Ledger | **SHA-256 hash chain** | Tamper-evident edge ledger + self-chained anchor log |
| 📄 Reports | **fpdf2** | One-click English PDF incident report (works offline) |
| 🔌 Backend | **FastAPI + WebSocket** | `/api/tracks`, alerts, chain verify — C2 integration contract |
| 🐍 Runtime | **Python 3.14, Torch 2.13 CPU** | Single-language stack, no GPU required |
| 🎨 Theme | **Light + Dark tactical** | Hybrid palette, pulsing LIVE indicators, tactical focus states |

---

## 🧪 Tests

```bash
python -m pytest          # 167 tests — pipeline contract, API, converters (~4 s)
```

The suite covers the pipeline contract (`Detection → Tracker → Fence → ANPR → HashChain`), every REST endpoint plus the WebSocket, and the dataset converters. Detections are stubbed, so the tests run in seconds and **download no model weights**.

Plus a headless Streamlit AppTest smoke suite (login gate, SOS, offline queue, anchors, PDF, alert console) and a YOLO `track()` smoke test with a persistent-ID assertion.

---

## 💰 Hardware Tiers

| Tier | Hardware | Cost | Runs |
|---|---|---|---|
| **Tier 1** — Check post | Jetson Orin Nano | ~$150–250 | Full AI: detect, track, ANPR, fence |
| **Tier 2** — Remote node | Microcontroller (MCU) | ~$20–30 | Motion trigger, store-and-forward |
| **Tier 3** — Command | Existing PC / browser | $0 | This dashboard |

---

## 🧭 Key Design Decisions

1. **Edge-first** — all inference runs locally; only compact metadata travels upstream (never video).
2. **Signal-loss-is-itself-an-alert** — a blinded camera triggers escalation, not silence.
3. **Offline-first** — link dies? AI keeps working, events queue locally, auto-sync on restore.
4. **Tamper-evident** — the SHA-256 hash chain makes the audit trail provably immutable.
5. **Honest labelling** — every non-live feed carries a visible `SIMULATED` tag; the risk formula and face-detection accuracy are documented, not inflated.
6. **CPU-only & budget-tiered** — no GPU required; a full post costs under ~$250, a remote node under ~$50.
7. **India-tuned** — autorickshaw class, Hindi action lines, IDD fine-tuning.

---

## 🎤 Pitch Materials

Presentation deck, demo script, recording guide, judging rubric answers and the full package live in [`pitch/`](pitch/).

---

## 🗣️ Field Doctrines

> *"The border never sleeps — and neither do we."*
> *"Every alert in time is a crisis avoided."*
> *"Vigilance is the price of every peaceful dawn."*
> *"Technology watches where eyes cannot reach."*
> *"सतर्क सीमा, सुरक्षित देश।"*

---

## 📜 License

Smart India Hackathon 2026 — Team BorderEye 🇮🇳
