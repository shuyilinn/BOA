"""FastAPI web application for BOA interactive interface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.run_manager import RunStatus, run_manager
from web.web_config import JUDGER_PRESETS, MODEL_PRESETS

app = FastAPI(title="BOA Web Interface")

# CORS: allow GitHub Pages and local dev to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your GitHub Pages URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    prompt: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    likelihood: float = 0.0001
    search_strategy: str = "greedy"
    pa_min_depth: int = 3
    pa_shallow_samples: int = 20
    pa_shallow_score: float = 8000
    pa_subtree_top_k: int = 5
    pa_search_temperature: float = 0.5
    mcts_exploration_constant: float = 1.414
    judger_type: str = "nuanced"  # nuanced | agent_safety | custom
    custom_judger_prompt: str = ""
    time_limit_sec: float = 120
    chunk_size: int = 2


class RunResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    """Return available target models with their default sampling params."""
    return {
        "models": [
            {"id": model_id, "defaults": defaults}
            for model_id, defaults in MODEL_PRESETS.items()
        ]
    }


@app.get("/api/judgers")
def list_judgers():
    """Return available judger presets."""
    return {"judgers": JUDGER_PRESETS}


@app.post("/api/run", response_model=RunResponse)
def start_run(req: RunRequest):
    """Start a new BOA run."""
    try:
        run_id = run_manager.start_run(
            prompt=req.prompt,
            model=req.model,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            likelihood=req.likelihood,
            search_strategy=req.search_strategy,
            pa_min_depth=req.pa_min_depth,
            pa_shallow_samples=req.pa_shallow_samples,
            pa_shallow_score=req.pa_shallow_score,
            pa_subtree_top_k=req.pa_subtree_top_k,
            pa_search_temperature=req.pa_search_temperature,
            mcts_exploration_constant=req.mcts_exploration_constant,
            judger_type=req.judger_type,
            custom_judger_prompt=req.custom_judger_prompt,
            time_limit_sec=req.time_limit_sec,
            chunk_size=req.chunk_size,
        )
        return RunResponse(run_id=run_id, status="started")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/run/{run_id}/stream")
async def stream_events(run_id: str):
    """SSE endpoint that streams tree updates and results."""
    state = run_manager.current_run
    if state is None or state.run_id != run_id:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        while True:
            # Non-blocking poll of the thread-safe queue
            try:
                event = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, state.event_queue.get, True, 0.5),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, Exception):
                # Check if this run is no longer current (replaced by a new run)
                if run_manager.current_run is not state:
                    break
                # Check if run finished
                if state.status in (RunStatus.COMPLETED, RunStatus.ERROR):
                    # Drain remaining events
                    while not state.event_queue.empty():
                        try:
                            event = state.event_queue.get_nowait()
                            yield f"event: {event['type']}\ndata: {json.dumps(event.get('data', {}), ensure_ascii=False)}\n\n"
                        except Exception:
                            break
                    break
                continue

            event_type = event.get("type", "unknown")
            event_data = event.get("data", {})
            yield f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            if event_type == "done":
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/run/{run_id}/status")
def get_run_status(run_id: str):
    """Get current run status."""
    state = run_manager.current_run
    if state is None or state.run_id != run_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": state.run_id,
        "status": state.status.value,
        "error": state.error,
    }


@app.post("/api/run/{run_id}/cancel")
def cancel_run(run_id: str):
    """Cancel a running job."""
    ok = run_manager.cancel_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found or not running")
    return {"status": "cancelling"}


class SampleRequest(BaseModel):
    n: int = 5


@app.post("/api/run/{run_id}/sample")
def sample_sequences(run_id: str, req: SampleRequest):
    """Sample N complete responses from the root node after a run completes."""
    state = run_manager.current_run
    if state is None or state.run_id != run_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if state.status == RunStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Run still in progress")
    try:
        samples = run_manager.sample_sequences(n=req.n)
        return {"samples": samples}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Static files / SPA
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return index_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
