"""
Exam Stress Balancer — Enhanced API (v3.0)
"""

import json, time
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from src.utils.logger import get_logger
from src.utils.helpers import load_pickle
from src.rl.agent.state_discretizer import discretize_state
from src.rl.student_environment import StudentEnvironment
from src.analytics.wellness import (
    compute_burnout_risk, compute_productivity_score,
    compute_wellness_score, classify_urgency,
    compute_adaptive_weights, get_recovery_suggestions,
)
from src.rl.planner.study_planner import generate_study_plan

logger = get_logger(__name__)

_model_store = {"q_table_q": None, "q_table_s": None, "loaded": False, "startup_time": None}
_SESSION_HISTORY: list[dict] = []
_HISTORY_FILE = Path("data/session_history.json")

ACTION_LABELS = {0: "Study", 1: "Revise", 2: "Sleep/Break"}
ACTION_ICONS  = {0: "📖", 1: "🔄", 2: "💤"}

_LEVEL_MAP = {
    "fatigue":   {"low": 20,  "medium": 50,  "high": 85},
    "stress":    {"low": 20,  "medium": 50,  "high": 85},
    "retention": {"low": 0.2, "medium": 0.5, "high": 0.85},
}


def load_models():
    base = Path("models")
    q_path, s_path = base / "q_learning.pkl", base / "sarsa.pkl"
    if not q_path.exists() or not s_path.exists():
        logger.warning("Model files missing — running in rule-based fallback mode")
        return
    _model_store["q_table_q"] = load_pickle(q_path)
    _model_store["q_table_s"] = load_pickle(s_path)
    _model_store["loaded"] = True
    logger.info("Both RL models loaded")


def load_history():
    global _SESSION_HISTORY
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _HISTORY_FILE.exists():
        try:
            _SESSION_HISTORY = json.loads(_HISTORY_FILE.read_text())
        except Exception:
            _SESSION_HISTORY = []


def save_history():
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(json.dumps(_SESSION_HISTORY[-50:], indent=2))
    except Exception as e:
        logger.warning(f"Could not persist history: {e}")


def hybrid_predict(state_dict: dict) -> tuple[int, dict]:
    actions = [0, 1, 2]
    if state_dict.get("stress", 50) >= 95:
        return 2, {"q_learning": {}, "sarsa": {}, "combined": {}, "q_weight": 0.0, "sarsa_weight": 1.0, "safety_override": True}

    if not _model_store["loaded"]:
        if state_dict.get("stress", 50) > 70 or state_dict.get("fatigue", 50) > 70:
            action = 2
        elif state_dict.get("fatigue", 50) < 40:
            action = 0
        else:
            action = 1
        return action, {"q_learning": {}, "sarsa": {}, "combined": {}, "q_weight": 0.6, "sarsa_weight": 0.4}

    q_w, s_w = compute_adaptive_weights(state_dict)
    d_state = discretize_state(state_dict)
    qv_q = _model_store["q_table_q"].get(d_state, {a: 0.0 for a in actions})
    qv_s = _model_store["q_table_s"].get(d_state, {a: 0.0 for a in actions})
    combined = {a: round(q_w * qv_q[a] + s_w * qv_s[a], 4) for a in actions}
    best = max(combined, key=combined.get)
    return best, {"q_learning": qv_q, "sarsa": qv_s, "combined": combined, "q_weight": q_w, "sarsa_weight": s_w}


def build_explanation(state: dict, action: int, debug: dict) -> str:
    stress, fatigue = state.get("stress", 50), state.get("fatigue", 50)
    retention, days_left = state.get("retention", 0.5), state.get("days_left", 7)
    q_w, s_w = debug.get("q_weight", 0.6), debug.get("sarsa_weight", 0.4)
    reasons = []

    if action == 0:
        reasons.append("You have enough cognitive bandwidth for new material right now.")
        if retention < 0.4:
            reasons.append(f"Your retention ({retention:.0%}) is low — active study builds the foundation you need.")
        if days_left <= 3:
            reasons.append(f"With only {days_left} days left, every study session has high impact.")
    elif action == 1:
        reasons.append("Revision consolidates existing knowledge without adding significant cognitive load.")
        if 0.4 <= retention <= 0.8:
            reasons.append(f"Your retention ({retention:.0%}) is in the sweet spot to benefit from revisiting material.")
        if stress > 40:
            reasons.append(f"Stress at {stress:.0f}/100 makes lighter revision more effective than heavy new study.")
    elif action == 2:
        if stress > 70:
            reasons.append(f"Stress is critically high ({stress:.0f}/100) — rest prevents burnout and memory loss.")
        if fatigue > 60:
            reasons.append(f"Fatigue at {fatigue:.0f}/100 severely limits learning efficiency.")
        reasons.append("Recovery now compounds into better performance over the remaining days.")

    if debug.get("safety_override"):
        reasons.append("Safety override active: Critical stress (>=95) detected — ignoring RL model and forcing immediate rest.")
    elif abs(q_w - 0.6) > 0.05:
        if s_w > q_w:
            reasons.append(f"Safety mode active (SARSA {s_w:.0%}): high fatigue/stress detected — conservative recommendations enabled.")
        else:
            reasons.append(f"Performance mode active (Q-Learning {q_w:.0%}): exam urgency detected — optimising for maximum output.")

    if debug.get("combined"):
        scores_str = ", ".join(f"{ACTION_LABELS[a]}={v:+.3f}" for a, v in sorted(debug["combined"].items()))
        reasons.append(f"Model scores: [{scores_str}].")
    return " ".join(reasons)


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Exam Stress Balancer API", version="3.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def startup():
    load_models()
    load_history()
    _model_store["startup_time"] = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({(time.time()-t0)*1000:.1f}ms)")
    return response


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, "history": _SESSION_HISTORY[-5:][::-1],
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    fatigue: str = Form(...), stress: str = Form(...),
    retention: str = Form(...), days_left: int = Form(...), difficulty: str = Form(...),
):
    fatigue_val   = _LEVEL_MAP["fatigue"].get(fatigue.lower(), 50)
    stress_val    = _LEVEL_MAP["stress"].get(stress.lower(), 50)
    retention_val = _LEVEL_MAP["retention"].get(retention.lower(), 0.5)
    state = {"fatigue": fatigue_val, "stress": stress_val, "retention": retention_val,
             "days_left": days_left, "difficulty": difficulty}

    action, debug  = hybrid_predict(state)
    explanation    = build_explanation(state, action, debug)
    burnout        = compute_burnout_risk(state)
    prod           = compute_productivity_score(state)
    wellness       = compute_wellness_score(state)
    urgency        = classify_urgency(days_left, retention_val)
    recovery       = get_recovery_suggestions(state, action)

    plan_state = dict(state, days_left=min(days_left, 14))
    study_plan = generate_study_plan(plan_state, hybrid_predict, max_days=min(days_left, 14))

    q_values = {ACTION_LABELS[a]: v for a, v in debug["combined"].items()} if debug.get("combined") else None

    session = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fatigue_label": fatigue, "stress_label": stress,
        "retention_label": retention, "days_left": days_left, "difficulty": difficulty,
        "action": ACTION_LABELS[action],
        "burnout_level": burnout.level, "burnout_score": burnout.score,
        "productivity_score": prod.score, "wellness_score": wellness.score,
    }
    _SESSION_HISTORY.append(session)
    save_history()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": ACTION_LABELS[action], "action_icon": ACTION_ICONS[action],
        "explanation": explanation, "q_values": q_values, "state": state,
        "burnout": burnout.to_dict(), "productivity": prod.to_dict(),
        "wellness": wellness.to_dict(), "urgency": urgency, "recovery_tips": recovery,
        "adaptive_weights": {"q": debug.get("q_weight", 0.6), "sarsa": debug.get("sarsa_weight", 0.4)},
        "study_plan": study_plan, "history": _SESSION_HISTORY[-5:][::-1],
    })


# ── JSON API ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    fatigue:    Optional[int] = Field(default=None, ge=0, le=100)
    stress:     Optional[int] = Field(default=None, ge=0, le=100)
    retention:  Optional[float] = Field(default=None, ge=0.0, le=1.0)
    days_left:  Optional[int] = Field(default=None, ge=0)
    difficulty: str   = Field(default="medium")

    # Client schema support
    stress_level: Optional[str] = None
    hours_studied: Optional[int] = None
    days_until_exam: Optional[int] = None
    current_performance: Optional[float] = None

    @model_validator(mode='before')
    @classmethod
    def map_client_schema(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if data.get('stress_level'):
                sm = {"low": 20, "medium": 50, "high": 85, "critical": 100}
                data['stress'] = sm.get(data['stress_level'].lower(), 50)
            
            if data.get('hours_studied') is not None:
                data['fatigue'] = min(data['hours_studied'] * 10, 100)
                
            if data.get('days_until_exam') is not None:
                data['days_left'] = data['days_until_exam']
                
            if data.get('current_performance') is not None:
                data['retention'] = data['current_performance']
                
            reqs = ['fatigue', 'stress', 'retention', 'days_left']
            for r in reqs:
                if data.get(r) is None:
                    raise ValueError(f"Missing required field: {r} (or mapping failed)")
                    
        return data


@app.get("/health")
def health():
    uptime = round(time.time() - _model_store["startup_time"], 1) if _model_store["startup_time"] else 0
    return {"status": "ok" if _model_store["loaded"] else "degraded",
            "models_loaded": _model_store["loaded"], "uptime_sec": uptime,
            "version": "3.0.0", "sessions_tracked": len(_SESSION_HISTORY)}


@app.post("/api/predict")
def api_predict(p: PredictRequest):
    state = p.model_dump()
    action, debug = hybrid_predict(state)
    return {"action_id": int(action), "recommended_action": ACTION_LABELS[action],
            "action_icon": ACTION_ICONS[action], "model": "hybrid_rl_v3",
            "adaptive_weights": {"q_learning": debug.get("q_weight", 0.6), "sarsa": debug.get("sarsa_weight", 0.4)},
            "debug": {"q_learning": debug["q_learning"], "sarsa": debug["sarsa"], "combined": debug["combined"]}}


@app.post("/api/explain")
def api_explain(p: PredictRequest):
    state = p.model_dump()
    action, debug = hybrid_predict(state)
    fmt = lambda d: {ACTION_LABELS[a]: round(v, 4) for a, v in d.items()}
    return {"action_id": int(action), "recommended_action": ACTION_LABELS[action],
            "model": "hybrid_rl_v3",
            "adaptive_weights": {"q_learning": debug.get("q_weight", 0.6), "sarsa": debug.get("sarsa_weight", 0.4)},
            "q_values": {"q_learning": fmt(debug["q_learning"]), "sarsa": fmt(debug["sarsa"]), "combined": fmt(debug["combined"])},
            "explanation": build_explanation(state, action, debug), "input_state": state}


@app.post("/api/study-plan")
def api_study_plan(p: PredictRequest):
    state = p.model_dump()
    max_days = min(state["days_left"], 14)
    plan = generate_study_plan(state, hybrid_predict, max_days=max_days)
    return {"input_state": state, "plan": plan["plan"], "summary": plan["summary"]}


@app.post("/api/burnout")
def api_burnout(p: PredictRequest):
    state = p.model_dump()
    return {"burnout_risk": compute_burnout_risk(state).to_dict(),
            "urgency": classify_urgency(state["days_left"], state["retention"]),
            "input_state": state}


@app.post("/api/wellness")
def api_wellness(p: PredictRequest):
    state = p.model_dump()
    return {"productivity": compute_productivity_score(state).to_dict(),
            "wellness": compute_wellness_score(state).to_dict(),
            "burnout_risk": compute_burnout_risk(state).to_dict(),
            "input_state": state}


@app.get("/api/history")
def api_history(limit: int = 20):
    return {"sessions": _SESSION_HISTORY[-limit:][::-1], "total": len(_SESSION_HISTORY)}


@app.delete("/api/history")
def api_clear_history():
    global _SESSION_HISTORY
    _SESSION_HISTORY = []
    save_history()
    return {"message": "History cleared", "status": "ok"}


@app.post("/simulate")
def simulate(payload: BaseModel):
    pass  # kept for compatibility


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
