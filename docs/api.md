# API Documentation

The Exam Stress Balancer exposes a FastAPI JSON service for integrating the RL logic into frontend applications.

## Dual-Schema Input Format

As of `v3.1.0`, all endpoints accept **either** the Internal ML Schema or the Client Schema.

**Client Schema (Recommended for external apps):**
```json
{
  "stress_level": "low", // "low", "medium", "high", "critical"
  "hours_studied": 4,    // int
  "days_until_exam": 7,  // int
  "current_performance": 0.5 // float 0.0-1.0
}
```

**Internal ML Schema (Legacy):**
```json
{
  "fatigue": 40,      // int 0-100
  "stress": 50,       // int 0-100
  "retention": 0.5,   // float 0.0-1.0
  "days_left": 7,     // int
  "difficulty": "medium" // "easy", "medium", "hard"
}
```

---

## `POST /api/explain`

The primary endpoint for generating an actionable recommendation with human-readable reasoning.

**Request:** `PredictRequest` (JSON body)

**Response:**
```json
{
  "action_id": 1,
  "recommended_action": "Revise",
  "model": "hybrid_rl_v3",
  "adaptive_weights": {
    "q_learning": 0.6,
    "sarsa": 0.4
  },
  "q_values": {
    "q_learning": {"Study": -12.7238, "Revise": 2.04, "Sleep/Break": -11.0},
    "sarsa": {"Study": -12.9149, "Revise": 2.04, "Sleep/Break": -11.0},
    "combined": {"Study": -12.8002, "Revise": 2.04, "Sleep/Break": -11.0}
  },
  "explanation": "Revision consolidates existing knowledge without adding significant cognitive load. Your retention (50%) is in the sweet spot to benefit from revisiting material. Stress at 50/100 makes lighter revision more effective than heavy new study.",
  "input_state": {
    "fatigue": 40,
    "stress": 50,
    "retention": 0.5,
    "days_left": 7,
    "difficulty": "medium"
  }
}
```

---

## `POST /api/predict`

A lightweight version of `/api/explain` that omits the human-readable explanation and formatting, useful for high-throughput headless agents.

**Request:** `PredictRequest` (JSON body)

**Response:**
```json
{
  "action_id": 1,
  "recommended_action": "Revise",
  "action_icon": "🔄",
  "model": "hybrid_rl_v3",
  "adaptive_weights": {"q_learning": 0.6, "sarsa": 0.4},
  "debug": { /* raw Q-values */ }
}
```

---

## `POST /api/study-plan`

Simulates the RL agent forward to generate a day-by-day plan.

**Request:** `PredictRequest` (JSON body)

**Response:**
```json
{
  "input_state": { ... },
  "plan": [
    {
      "day": 1,
      "days_remaining": 7,
      "action_id": 0,
      "action": "Study",
      "action_icon": "📖",
      "action_color": "#6366f1",
      "advice": "Focus on your weakest topics first while your mind is fresh.",
      "confidence": 85.0,
      "state_snapshot": {
        "fatigue": 40.0,
        "stress": 50.0,
        "retention": 0.5
      },
      "burnout_level": "LOW",
      "burnout_score": 15,
      "productivity_score": 75,
      "productivity_label": "High",
      "wellness_score": 80,
      "reward_estimate": 10.5
    }
  ],
  "summary": "Focus heavily on Study early on, followed by revision."
}
```

---

## `GET /health`

Returns the operational status of the API and RL models.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "uptime_sec": 3640.2,
  "version": "3.1.0",
  "sessions_tracked": 12
}
```

---

## Error Codes

- `200 OK`: Request succeeded.
- `422 Unprocessable Entity`: Input validation failed. Ensure you are sending the correct data types and that fields are not missing (you must provide either the full ML schema or the full Client schema).
- `500 Internal Server Error`: RL models failed to load or evaluate.
