# Exam Stress Balancer — Hybrid RL + MLOps (v3.1.0)

> **AI-powered adaptive exam preparation assistant** using Reinforcement Learning to optimize study balance, prevent burnout, and maximize exam readiness.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

---

## 🎯 What It Does

The Exam Stress Balancer is designed for students struggling to manage their time, stress, and energy leading up to exams. It analyzes a student's cognitive and temporal state and generates:
- **Immediate actionable advice:** Study, Revise, or Sleep/Break.
- **Explainable AI Reasoning:** Human-readable explanations mapping the RL model's weights to practical reality.
- **Multi-day Adaptive Study Plans:** Forward-simulated schedules optimized for peak readiness.
- **Wellness & Productivity Scores:** Composite health indicators.

## 🧠 Hybrid RL Architecture

The project employs a dual-agent Reinforcement Learning model to balance aggressive optimization with psychological safety:
1. **Q-Learning Agent (Off-policy):** Optimizes for maximum knowledge acquisition.
2. **SARSA Agent (On-policy):** Takes a conservative, burnout-preventing approach.

**Adaptive Weighting Engine:** Depending on the student's state, the engine shifts weights (e.g., 75% Q-Learning for high-urgency/low-retention, or 55% SARSA for high-fatigue).
**Critical Safety Override (v3.1.0):** If stress levels hit a critical threshold (>=95), the RL models are completely overridden, forcing immediate rest.

*For full technical details, see [docs/architecture.md](docs/architecture.md).*

---

## 🚀 API Endpoints

The API natively supports both a dedicated ML schema and an intuitive client schema via dual Pydantic validation.

**Example Request to `/api/explain`:**
```bash
curl -X POST https://exam-stresss-balancer.onrender.com/api/explain \
  -H "Content-Type: application/json" \
  -d '{"stress_level": "high", "hours_studied": 6, "days_until_exam": 3, "current_performance": 0.4}'
```

**Example Response:**
```json
{
  "action_id": 1,
  "recommended_action": "Revise",
  "adaptive_weights": {"q_learning": 0.45, "sarsa": 0.55},
  "explanation": "Revision consolidates existing knowledge without adding significant cognitive load. Safety mode active (SARSA 55%): high fatigue/stress detected — conservative recommendations enabled."
}
```

*For complete endpoint documentation, see [docs/api.md](docs/api.md).*

---

## 💻 Local Development

1. **Clone & Setup:**
   ```bash
   git clone <repo-url> && cd exam-stress-balancer-mlops
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Train Models (DVC + MLflow):**
   Modify `params.yaml`, then run the reproducible pipeline:
   ```bash
   dvc repro
   ```
   *View metrics:* `mlflow ui --port 5000`
3. **Run API:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

## 🐳 Docker Setup
```bash
docker-compose up --build
# App is live at http://localhost:8000
```

## ☁️ Render Deployment
The project is configured for PaaS deployment (e.g., Render, Railway) via the included `Procfile`. Render auto-deploys upon push to the main branch. Ensure you set the `PORT=8000` environment variable.

---

## 📋 Changelog
For version history, see the [CHANGELOG.md](CHANGELOG.md).

*Built with Python 3.11 · FastAPI · MLflow · DVC · Docker*
