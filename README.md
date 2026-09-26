<div align="center">

#  VOX
### Offline Hinglish Delivery Assistant — On-Device NLU for Low-Connectivity Environments

*A production-grade, offline-first Natural Language Understanding system that understands Hinglish voice/text commands — entirely on-device, zero internet required.*

[Overview](#-overview) · [Architecture](#-system-architecture) · [Benchmarks](#-benchmarks) · [Getting Started](#-getting-started) · [API Reference](#-api-reference) · [Roadmap](#-roadmap)

</div>

---

##  Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Supported Intents](#-supported-intents)
- [ML Pipeline](#-ml-pipeline)
- [Backend API](#-backend-api)
- [Frontend Dashboard](#-frontend-dashboard)
- [Docker Setup](#-docker-setup)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Benchmarks](#-benchmarks)
- [Roadmap](#-roadmap)
- [Tech Stack](#-tech-stack)

---

##  Overview

**VOX** is a fully offline Natural Language Understanding (NLU) engine purpose-built for last-mile delivery partners operating in poor or zero-connectivity zones across India. It parses **Hinglish** (Hindi + English code-mixed) voice/text commands using a custom, lightweight **Bidirectional GRU** model exported to **ONNX**, enabling real-time inference on low-end Android hardware (2GB–4GB RAM, CPU-only) — **no cloud dependency at any point.**

### Why VOX?
Millions of delivery workers in Tier-2/Tier-3 India face inconsistent connectivity, and mainstream voice assistants both require internet and fail to understand code-mixed Hinglish. VOX closes that gap: a model small enough to ship on-device (**~2.1MB**), fast enough to feel instant (**sub-10ms inference**), and accurate enough to trust in production (**~95%+ intent accuracy, 100% slot precision**).

> The project evolved through two prior iterations (EdgeAssist → ONYX → VOX), converging on a hybrid **ML + deterministic-regex** architecture that pairs a learned intent classifier with a rule-based slot extractor for guaranteed-precision entity extraction — a deliberate design choice over an end-to-end neural pipeline, explained below.

---

##  Key Features

| Feature | Description |
|---|---|
| **Hinglish NLU** | Understands natural code-mixed Hindi-English commands out of the box |
| **Sub-10ms Inference** | Bi-GRU model under 300K parameters — fast even on budget hardware |
| **Dual Extraction** | Intent classification (ML) + slot extraction (deterministic regex engine) |
| **Benchmark Suite** | Built-in `/benchmark` endpoint for latency, memory, and accuracy profiling |
| **Modular Design** | Clean separation: ML pipeline → ONNX inference → FastAPI → React UI |
| **Docker Support** | Fully containerized backend and frontend via Docker Compose |

---

##  System Architecture

```
                     ┌───────────────────────────────┐
                     │       DELIVERY PARTNER          │
                     │    (Text / Voice Command)       │
                     └────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────┐
                     │   REACT FRONTEND (Vite)         │
                     │  Mobile-first UI · Voice        │
                     │  Recorder · Intent Visualizer   │
                     └────────────────┬─────────────────┘
                                       │  HTTP (local)
                                       ▼
                     ┌───────────────────────────────────────────────┐
                     │               FASTAPI BACKEND                  │
                     │  ┌────────────────┐   ┌──────────────────────┐│
                     │  │  ONNX Runtime   │   │ Rule-Based Slot       ││
                     │  │  (CPU only)     │   │ Extractor (Regex,     ││
                     │  │  Bi-GRU, ~300K  │   │ 100% Precision)       ││
                     │  │  params         │   │ delay_time · order_ref││
                     │  │                 │   │ customer_status ·     ││
                     │  │                 │   │ reason                ││
                     │  └────────────────┘   └──────────────────────┘│
                     └────────────────┬─────────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────┐
                     │           ML PIPELINE           │
                     │  data_generator → tokenizer →   │
                     │  train → export (.onnx)         │
                     └───────────────────────────────┘
```

**Design rationale:** the intent classifier is a learned model (handles linguistic variation, code-mixing, phrasing drift), while slot extraction is deliberately kept **deterministic regex, not neural** — for a delivery-ops use case, mis-extracting a delay time or order reference is far costlier than the marginal recall a neural slot-filler would add. This hybrid split trades a small amount of extraction flexibility for **100% precision on structured fields**.

---

##  Project Structure

```
VOX/
│
├── backend/                          # FastAPI Inference Server
│   ├── api/
│   │   └── routes.py                   # All API endpoint definitions
│   ├── core/
│   │   ├── exceptions.py               # Custom exception handlers
│   │   └── logger.py                   # Structured logging config
│   ├── schemas/
│   │   └── predict.py                  # Pydantic request/response models
│   ├── services/
│   │   ├── inference_service.py        # ONNX model loading & prediction
│   │   ├── slot_extractor.py           # Regex-based slot extraction engine
│   │   └── response_generator.py       # Contextual response builder
│   ├── main.py                         # FastAPI app entrypoint
│   └── Dockerfile                      # Backend container definition
│
├── data/                             # Datasets
│   ├── full_dataset.csv                # Complete labeled Hinglish corpus
│   ├── train.csv                       # Training split (80%)
│   ├── val.csv                         # Validation split (10%)
│   └── test.csv                        # Held-out test split (10%)
│
├── ml_pipeline/                      # Training & Export Pipeline
│   ├── data_generator.py               # Synthetic Hinglish data generation
│   ├── dataset.py                      # PyTorch Dataset class
│   ├── tokenizer.py                    # Word-level tokenizer with <OOV>
│   ├── model.py                        # Bi-GRU architecture (PyTorch)
│   ├── train.py                        # Training loop + early stopping
│   ├── evaluate.py                     # Evaluation & metrics reporting
│   └── inference.py                    # Local inference test script
│
├── model/                            # Exported Model Artifacts
│   ├── best_model.pth                  # Best PyTorch checkpoint
│   ├── quantized_model.pth             # Quantized model (optional)
│   ├── model.onnx                      # Production ONNX model
│   ├── model.onnx.data                 # External ONNX data (if applicable)
│   └── vocab.json                      # Word-to-index vocabulary map
│
├── frontend/                         # React 18 + TypeScript (Vite)
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/
│   │   ├── App.tsx                     # Root component
│   │   ├── App.css                     # Global styles (Vanilla CSS)
│   │   ├── main.tsx                    # Vite entrypoint
│   │   └── index.css                   # Base CSS reset & variables
│   └── Dockerfile                      # Frontend container definition
│
├── docker-compose.yaml               # Multi-container orchestration
└── .dockerignore                     # Files excluded from Docker build context
```

---

##  Supported Intents

VOX classifies delivery partner commands into **5 core intents**:

| Intent | Example Hinglish Command | Description |
|---|---|---|
| `get_address` | *"Bhai next order ka address batao"* | Fetch delivery address details |
| `report_delay` | *"Traffic ki wajah se 10 min late honga"* | Report ETA delay with reason |
| `order_issue` | *"Packet damage ho gaya hai"* | Flag a problem with the order |
| `customer_unavailable` | *"Customer phone nahi utha raha"* | Mark customer as unreachable |
| `navigation_help` | *"Map stuck ho gaya location do"* | Request navigation assistance |

**Extracted Slots (example):**
```json
{
  "intent": "report_delay",
  "confidence": 0.96,
  "slots": {
    "delay_time": "10 min",
    "delay_reason": "Traffic"
  }
}
```

---

##  ML Pipeline

### Model Architecture
```
Input (20 tokens) → Embedding (64-dim) → Bi-GRU (64 hidden × 2 directions)
                  → Global Max Pooling → Dense (128) → Output (5 classes)

Total Parameters: ~250,000   (well under a 1M-parameter budget for mobile deployment)
```

### Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Max Sequence Length | 20 tokens |
| Embedding Dimension | 64 |
| GRU Hidden Size | 64 (bidirectional → 128) |
| Early Stopping | Enabled |
| OOV Token | `<OOV>` |
| Export Format | ONNX (dynamic batch axis) |

### Training the Model
```bash
# Step 1: Generate synthetic Hinglish dataset
python ml_pipeline/data_generator.py

# Step 2: Train the Bi-GRU model
python ml_pipeline/train.py

# Step 3: Evaluate on test set
python ml_pipeline/evaluate.py

# Step 4: Run local inference test
python ml_pipeline/inference.py
```

---

##  Backend API

### Prerequisites
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start Server
```bash
uvicorn main:app --reload --port 8000
```

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — model load status |
| `POST` | `/predict` | Run NLU on a Hinglish text command |
| `POST` | `/voice` | Accept audio file, run STT → predict |
| `POST` | `/benchmark` | Batch inference with latency & accuracy metrics |

---

##  Frontend Dashboard

A mobile-first, dark-mode dashboard that simulates the delivery partner's Android interface.

**Features:**
- **Text Input** — type any Hinglish command
- **Intent Gauge** — visual confidence meter for the classified intent
- **Slot Chips** — extracted entities rendered as interactive badges
- **Response Panel** — contextual action suggestions based on intent

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```
App runs at `http://localhost:5173`

---

##  Docker Setup

VOX ships fully containerized — separate `Dockerfile`s for backend and frontend, orchestrated by a root-level `docker-compose.yaml`.

### Prerequisites
- Docker 24+ installed and running
- Docker Compose v2 (ships with Docker Desktop; on Linux use `docker compose`, not `docker-compose`)

### Start All Services
```bash
docker compose up --build
```
This builds and starts:
- **FastAPI backend** → `http://localhost:8000`
- **React frontend** → `http://localhost:5173`

### Start in Detached Mode
```bash
docker compose up --build -d
```

### Stop All Services
```bash
docker compose down
```

### Build Individual Services
```bash
# Backend only
docker build -t vox-backend ./backend

# Frontend only
docker build -t vox-frontend ./frontend
```

### Run Individual Containers
```bash
# Backend
docker run -p 8000:8000 vox-backend

# Frontend
docker run -p 5173:5173 vox-frontend
```

> **Notes:**
> - `.dockerignore` excludes build artifacts, virtual environments, `node_modules`, and model checkpoints from the build context to keep images minimal.
> - The ONNX model and `vocab.json` in `/model` must exist before building the backend image — run the ML pipeline locally first, or mount `/model` as a volume.
> - If you change the backend API URL, update `VITE_API_URL` in `docker-compose.yaml`.

---

##  Getting Started

### Option 1: Run with Docker (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/Priyankshu-07/VOX.git
cd VOX

# 2. Train and export the model (required before Docker build)
python ml_pipeline/data_generator.py
python ml_pipeline/train.py

# 3. Start all services
docker compose up --build
```
Backend: `http://localhost:8000` · Frontend: `http://localhost:5173`

### Option 2: Run Manually

**1. Clone the repository**
```bash
git clone https://github.com/Priyankshu-07/VOX.git
cd VOX
```

**2. Train & export the model**
```bash
python ml_pipeline/data_generator.py
python ml_pipeline/train.py
# Model artifacts saved to /model/
```

**3. Start the backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**4. Start the frontend**
```bash
cd frontend
npm install && npm run dev
```

**5. Test a prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Traffic ki wajah se 10 min late honga"}'
```

**Expected response:**
```json
{
  "intent": "report_delay",
  "confidence": 0.96,
  "slots": {
    "delay_time": "10 min",
    "delay_reason": "Traffic"
  }
}
```

---

##  API Reference

### `POST /predict`
**Request:**
```json
{
  "text": "Customer phone nahi utha raha, order wapas leke aun kya?"
}
```
**Response:**
```json
{
  "intent": "customer_unavailable",
  "confidence": 0.94,
  "slots": {
    "customer_status": "phone nahi utha raha"
  }
}
```

### `POST /benchmark`
**Request:**
```json
{
  "texts": [
    "Bhai address galat hai",
    "Map stuck ho gaya location do",
    "20 min late hounga bhai"
  ]
}
```
**Response:**
```json
{
  "avg_latency_ms": 4.7,
  "memory_mb": 38.2,
  "accuracy": 0.96,
  "total_samples": 3
}
```

---

##  Benchmarks

Tested on CPU-only execution (Intel Core i5, single-threaded — emulating mobile constraints):

| Metric | Value |
|---|---|
| Avg Inference Latency | **< 10ms** per request |
| Model Size (ONNX) | **~2.1MB** |
| Peak Memory Usage | **~40MB** |
| Intent Classification Accuracy | **~95%+** on held-out test set |
| Slot Extraction Precision | **100%** (deterministic regex) |

> These benchmarks target emulation of low-end Android hardware (2GB–4GB RAM, ARM CPU). Native Android deployment may vary.

---

##  Roadmap

- [x] Synthetic Hinglish dataset generation (5 intents)
- [x] Bi-GRU model training + ONNX export
- [x] FastAPI inference server with slot extractor
- [x] React dashboard (dark mode, glassmorphism UI)
- [x] Docker support (Dockerfiles + Docker Compose)
- [ ] Integrate lightweight offline STT (Vosk / Whisper-tiny)
- [ ] Quantized INT8 ONNX model for further size reduction
- [ ] Native Android (Kotlin + ONNX Runtime Mobile) deployment
- [ ] Expand to 10+ intents with regional dialect support
- [ ] Over-the-air (OTA) vocabulary + model updates via delta patches

---

##  Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x |
| Inference Runtime | ONNX Runtime (CPU) |
| Backend | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Vanilla CSS (mobile-first, dark mode) |
| Containerization | Docker + Docker Compose |
| Target Platform | Android (2GB–4GB RAM, CPU-only) |

---

<div align="center">

**Built to prove that production-grade NLU doesn't need the cloud — just careful engineering under tight constraints.**

</div>
