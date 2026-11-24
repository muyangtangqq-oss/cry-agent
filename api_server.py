"""FastAPI server — exposes cry-agent as a REST API.

Endpoints:
  POST /api/classify       — upload audio file, get judgment
  POST /api/feedback       — submit parent feedback (correct/wrong)
  GET  /api/profile        — get baby profile + signal weights
  GET  /api/history        — get cry history
  GET  /api/stats          — get accuracy stats
  WS   /ws/live            — WebSocket for real-time results
"""
import os
import time
import json
import tempfile
from typing import Optional

from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import load_config
from pipeline import CryAgentPipeline
from baby_profile import BabyProfile
from audio_inference import CryAudioClassifier

app = FastAPI(title="Cry-Agent API", version="0.1.0")

# Global state
cfg = load_config()
pipeline = CryAgentPipeline(cfg)
audio_clf = CryAudioClassifier()
profile = BabyProfile.load(os.path.expanduser("~/.config/cry-agent/profile.json"))
profile_path = os.path.expanduser("~/.config/cry-agent/profile.json")


class ClassifyRequest(BaseModel):
    postures: list[str] = []
    face_state: Optional[str] = None
    room_temp: Optional[float] = None
    ambient_noise_db: Optional[float] = None
    feed_gap_hours: float = 2.0
    sleep_gap_hours: float = 3.0
    age_days: int = 42


class FeedbackRequest(BaseModel):
    event_index: int
    actual_cause: str


@app.post("/api/classify")
async def classify_audio(
    file: UploadFile = File(...),
    postures: str = Form(""),
    face_state: str = Form(None),
    room_temp: float = Form(None),
    noise_db: float = Form(None),
    feed_gap: float = Form(2.0),
    sleep_gap: float = Form(3.0),
    age_days: int = Form(42),
):
    """Upload audio file and get cry cause judgment."""
    # Save uploaded file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    # Update profile
    profile.age_days = age_days
    profile.last_feed_time = time.time() - feed_gap * 3600
    profile.last_sleep_end = time.time() - sleep_gap * 3600

    posture_list = [p.strip() for p in postures.split(",") if p.strip()]

    # Run pipeline
    result = pipeline.run(
        audio_path=tmp.name,
        postures=posture_list,
        face_state=face_state,
        room_temp=room_temp,
        ambient_noise_db=noise_db,
        profile=profile,
    )

    os.unlink(tmp.name)

    # Record event
    from baby_profile import CryEvent
    event_idx = len(profile.cry_history)
    profile.record_cry(CryEvent(
        timestamp=time.time(),
        cause=result.cause,
        confidence=result.confidence,
        signals={"audio": result.audio_cause, "vision": posture_list},
    ))
    profile.save(profile_path)

    return {
        "cause": result.cause,
        "confidence": result.confidence,
        "model": result.model,
        "audio_cause": result.audio_cause,
        "audio_confidence": result.audio_confidence,
        "fused_cause": result.fused_cause,
        "fused_confidence": result.fused_confidence,
        "escalated": result.escalated,
        "critical_override": result.critical_override,
        "audio_ood": result.audio_ood,
        "session_ood": result.session_ood,
        "audio_ood_score": result.audio_ood_score,
        "trusted": result.trusted,
        "latency_ms": result.latency_ms,
        "all_scores": result.all_scores,
        "event_index": event_idx,
    }


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit parent feedback to adjust signal weights."""
    profile.record_feedback(req.event_index, req.actual_cause)
    profile.save(profile_path)
    return {
        "status": "ok",
        "signal_weights": profile.signal_weights,
        "library_size": len(profile.cry_history),
    }


@app.get("/api/profile")
async def get_profile():
    """Get baby profile and current signal weights."""
    return {
        "age_days": profile.age_days,
        "signal_weights": profile.signal_weights,
        "cry_count": len(profile.cry_history),
        "misjudgment_notes": profile.misjudgment_notes[-5:],
        "feed_interval_mean": profile.feed_interval_mean_hours,
    }


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent cry events."""
    history = profile.cry_history[-limit:]
    return {"events": history, "total": len(profile.cry_history)}


@app.get("/api/stats")
async def get_stats():
    """Get accuracy statistics."""
    total = len(profile.cry_history)
    feedback_count = sum(1 for e in profile.cry_history if e.get("feedback"))
    correct = sum(1 for e in profile.cry_history
                  if e.get("feedback") and e["feedback"] == e["cause"])

    return {
        "total_events": total,
        "feedback_count": feedback_count,
        "correct_count": correct,
        "accuracy": correct / max(feedback_count, 1),
        "signal_weights": profile.signal_weights,
    }


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """WebSocket for real-time results (for dashboard live mode)."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "classify":
                # Expect base64 audio or file path
                # For simplicity, expect a file path
                audio_path = msg.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    profile.last_feed_time = time.time() - msg.get("feed_gap", 2.0) * 3600
                    profile.last_sleep_end = time.time() - msg.get("sleep_gap", 3.0) * 3600

                    result = pipeline.run(
                        audio_path=audio_path,
                        postures=msg.get("postures", []),
                        face_state=msg.get("face_state"),
                        room_temp=msg.get("room_temp"),
                        ambient_noise_db=msg.get("noise_db"),
                        profile=profile,
                    )
                    await ws.send_json({
                        "type": "result",
                        "cause": result.cause,
                        "confidence": result.confidence,
                        "model": result.model,
                        "latency_ms": result.latency_ms,
                    })
    except Exception:
        pass


# Serve dashboard
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


@app.get("/")
async def root():
    return HTMLResponse("""
    <html><body>
    <h1>Cry-Agent API</h1>
    <p><a href="/docs">API Docs</a></p>
    <p><a href="/dashboard">Dashboard</a></p>
    </body></html>
    """)
