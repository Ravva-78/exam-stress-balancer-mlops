# Exam Stress Balancer — Hybrid RL + MLOps (v3.0)

> **AI-powered adaptive exam preparation assistant** using Reinforcement Learning to optimise study balance, prevent burnout, and maximise exam readiness.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

---

## What It Does

Analyses a student's current cognitive state (fatigue, stress, retention, exam proximity) and generates:

- **Immediate RL recommendation** — Study / Revise / Sleep/Break
- **Multi-day adaptive study plan** — day-by-day schedule simulated forward by the hybrid RL model
- **Burnout risk scoring** — LOW / MEDIUM / HIGH / CRITICAL with root-cause factors
- **Productivity & Wellness scores** — composite health indicators
- **Explainable AI reasoning** — human-readable rationale with adaptive weight transparency
- **Session history tracking** — trend monitoring across sessions

---

## Architecture

```
Student State (5D)
  ├── fatigue ──────┐
  ├── stress        │    State Discretizer
  ├── retention ────┼──► (3-level binning) ──► Discrete State Key
  ├── days_left     │
  └── difficulty ───┘

Q-Learning Agent ──────► Q-Values (off-policy, optimal)
SARSA Agent ──────────►  Q-Values (on-policy, safe)
                                 │
            Adaptive Weighting ◄─┘
            (state-aware fusion)
                   │
           Hybrid Recommendation
                   │
    ┌──────────────┴──────────────┐
    │                             │
Immediate Action          Multi-Day Plan
(Study / Revise /         (forward simulation,
 Sleep/Break)              up to 14 days)
```

### Adaptive Hybrid Weighting (v3.0)

Unlike the fixed 60/40 split in v2, weights now shift dynamically:

| Condition | Q-Learning | SARSA | Rationale |
|-----------|-----------|-------|-----------|
| High stress or fatigue (>70) | 45% | 55% | Safety mode: conservative, burnout-aware |
| Critical urgency + low retention | 75% | 25% | Performance mode: aggressive optimisation |
| Default | 60% | 40% | Balanced baseline |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `POST` | `/predict` | Form → full HTML dashboard result |
| `POST` | `/api/predict` | JSON single recommendation |
| `POST` | `/api/explain` | Full explainability + Q-values |
| `POST` | `/api/study-plan` | Multi-day adaptive plan (JSON) |
| `POST` | `/api/burnout` | Burnout risk analysis |
| `POST` | `/api/wellness` | Productivity + wellness scores |
| `GET` | `/api/history` | Session history |
| `DELETE` | `/api/history` | Clear history |
| `GET` | `/health` | Health + uptime check |
| `GET` | `/docs` | Swagger UI |

### Example — Study Plan Request

```bash
curl -X POST http://localhost:8000/api/study-plan \
  -H "Content-Type: application/json" \
  -d '{"fatigue":60,"stress":65,"retention":0.4,"days_left":7,"difficulty":"hard"}'
```

---

## State Space

| Feature | Type | Range | Bins |
|---------|------|-------|------|
| Fatigue | int | 0–100 | LOW / MEDIUM / HIGH |
| Stress | int | 0–100 | LOW / MEDIUM / HIGH |
| Retention | float | 0.0–1.0 | LOW / MEDIUM / HIGH |
| Days Left | int | 0–60 | HIGH / MEDIUM / LOW urgency |
| Difficulty | str | easy/medium/hard | EASY / MEDIUM / HARD |

State space size: **3 × 3 × 3 × 3 × 3 = 243** discrete states

## Action Space

| ID | Action | When Recommended |
|----|--------|-----------------|
| 0 | Study | Low fatigue, low/medium stress, low retention |
| 1 | Revise | Moderate states, medium-high retention |
| 2 | Sleep/Break | High fatigue OR high stress, or recovery needed |

---

## Reward Engineering

```python
# Study action
reward = learning_gain × 20 - 0.04 × fatigue - 0.03 × stress
if retention < 0.4:  reward += 5   # bonus: knowledge gap exists
if retention > 0.9:  reward -= 5   # penalty: over-studying

# Revise action  
reward = 0.06 × 14 - 0.02 × fatigue
if 0.4 ≤ retention ≤ 0.8:  reward += 2  # sweet spot bonus

# Break/Sleep action
reward = 1.0
if fatigue > 60 or stress > 60:  reward += 5  # justified rest

# Universal penalties
if fatigue > 85 and stress > 85:  reward -= 5   # double-danger
if fatigue > 85 or stress > 85:   reward -= 12  # burnout zone
if action == Study in burnout zone: reward -= 8  # extra penalty
```

---

## MLOps Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Experiment tracking | MLflow | Metrics, params, artifacts |
| Pipeline versioning | DVC | Reproducible train→eval pipeline |
| Containerisation | Docker + Compose | Portable deployment |
| API serving | FastAPI | REST + HTML UI |
| Model serialisation | Pickle | Q-table persistence |
| Visualisation | Matplotlib | Reward curves, state analysis |
| Logging | Python logging | Structured request/training logs |

### DVC Pipeline

```
dvc repro
```

Stages: `train → evaluate → compare_baselines → visualize`

---

## Quick Start

### Docker (recommended)

```bash
docker-compose up --build
# Open http://localhost:8000
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train both RL agents
python -m src.training.train_rl   # train Q-Learning
# Edit params.yaml: algorithm: sarsa
python -m src.training.train_rl   # train SARSA

# Run API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

---

## Project Structure

```
exam-stress-balancer/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI app (v3.0)
│   ├── analytics/
│   │   └── wellness.py          # Burnout, productivity, wellness scoring
│   ├── rl/
│   │   ├── agent/
│   │   │   ├── q_learning_agent.py
│   │   │   ├── sarsa_agent.py
│   │   │   └── state_discretizer.py
│   │   ├── planner/
│   │   │   └── study_planner.py # Multi-day plan generator
│   │   ├── student_environment.py
│   │   └── training/
│   ├── training/
│   │   └── train_rl.py
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── templates/index.html          # Dashboard frontend
├── static/style.css              # Dark dashboard CSS
├── models/                       # Trained Q-tables (.pkl)
├── data/session_history.json     # Session persistence
├── params.yaml                   # DVC-controlled hyperparams
├── dvc.yaml                      # Pipeline definition
├── docker-compose.yml
└── README.md
```

---

## Enhancements in v3.0

| Feature | v2 | v3 |
|---------|----|----|
| RL recommendation | Single action | Single action + multi-day plan |
| Hybrid weights | Fixed 60/40 | Adaptive (state-aware) |
| Burnout analysis | Warning banner only | Scored risk + root causes + recovery tip |
| Productivity scoring | ✗ | ✓ Yerkes-Dodson model |
| Wellness scoring | ✗ | ✓ Composite fatigue/stress/retention |
| Session history | ✗ | ✓ Persistent JSON + UI display |
| Study plan | ✗ | ✓ Day-by-day RL simulation |
| Urgency classification | ✗ | ✓ 4-level + retention warning |
| Recovery suggestions | ✗ | ✓ Contextual 2–4 tips |
| Frontend | Glassmorphism form | Dark analytics dashboard |
| API endpoints | 5 | 10 |

---

## Deployment

### Railway / Render

```bash
# Procfile already configured
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

Set environment variable: `PORT=8000`

### Environment Variables

```bash
cp .env.example .env
# MLFLOW_TRACKING_URI=http://localhost:5000
# API_HOST=0.0.0.0
# API_PORT=8000
```

---

## Research Context

This project demonstrates:

1. **Hybrid RL design** — combining off-policy (Q-Learning) and on-policy (SARSA) algorithms with adaptive fusion for safer, more robust recommendations
2. **Human-centered RL reward engineering** — balancing learning efficiency with burnout prevention
3. **MLOps best practices** — experiment tracking, pipeline versioning, containerised deployment
4. **Explainable AI** — transparent model reasoning with adaptive weight disclosure
5. **Forward simulation planning** — using trained RL policy to generate multi-step plans

---

*Built with Python 3.11 · FastAPI · MLflow · DVC · Docker*
