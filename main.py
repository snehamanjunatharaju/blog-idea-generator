import csv
import io
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from idea_engine import ProviderError, generate_ideas, provider_status, regenerate_one
from run_history import metrics_summary, record_run

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("blog_idea_generator")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

REQUEST_LIMIT = int(os.getenv("REQUESTS_PER_MINUTE", "10"))
request_times: dict[str, deque[float]] = defaultdict(deque)
request_lock = Lock()
TONES = {"professional", "conversational", "humorous", "inspirational", "educational"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        logger.info("provider_status=%s", json.dumps(provider_status()))
    except ProviderError as exc:
        logger.error("Provider configuration error: %s", exc)
    yield


app = FastAPI(title="Blog Idea Generator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with request_lock:
        recent = request_times[client_ip]
        while recent and now - recent[0] >= 60:
            recent.popleft()
        if len(recent) >= REQUEST_LIMIT:
            raise HTTPException(status_code=429, detail="Request limit reached. Try again in a minute.")
        recent.append(now)


def _validate_fields(niche: str, audience: str, tone: str) -> None:
    if not niche.strip():
        raise HTTPException(status_code=422, detail="Niche cannot be blank.")
    if not audience.strip():
        raise HTTPException(status_code=422, detail="Audience cannot be blank.")
    if tone not in TONES:
        raise HTTPException(status_code=422, detail="Choose a supported tone.")


def _response(result) -> dict:
    return {
        "ideas": result.ideas,
        "sources": result.sources,
        "mode": result.mode,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "validation_pass_rate": result.validation_pass_rate,
        "duplicate_rate": result.duplicate_rate,
    }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        mode = provider_status()
    except ProviderError:
        mode = {"mode": "configuration error", "model": ""}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"mode": mode},
    )


@app.get("/mode")
async def get_mode():
    try:
        return provider_status()
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/generate")
async def generate(
    request: Request,
    niche: str = Form(..., max_length=120),
    audience: str = Form("freshers in India", max_length=120),
    num_ideas: int = Form(3, ge=1, le=6),
    tone: str = Form("professional"),
    model: str = Form(""),
):
    _check_rate_limit(request)
    _validate_fields(niche, audience, tone)
    started = time.perf_counter()
    try:
        result = await generate_ideas(niche.strip(), audience.strip(), num_ideas, tone, model or None)
    except (ProviderError, httpx.HTTPError, KeyError) as exc:
        logger.exception("Idea generation failed")
        raise HTTPException(status_code=502, detail=f"Idea generation failed: {exc}") from exc
    except (ValidationError, ValueError) as exc:
        logger.exception("Provider returned invalid idea data")
        raise HTTPException(status_code=502, detail="Provider returned invalid idea data.") from exc

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    record_run(
        niche=niche.strip(),
        audience=audience.strip(),
        tone=tone,
        mode=result.mode,
        model=result.model,
        latency_ms=elapsed,
        ideas=result.ideas,
        duplicate_rate=result.duplicate_rate,
    )
    logger.info(
        "generation_completed mode=%s model=%s count=%d latency_ms=%.2f pass_rate=%.2f duplicate_rate=%.2f",
        result.mode, result.model, len(result.ideas), elapsed, result.validation_pass_rate, result.duplicate_rate,
    )
    return _response(result)


@app.post("/regenerate")
async def regenerate(
    request: Request,
    niche: str = Form(..., max_length=120),
    audience: str = Form("freshers in India", max_length=120),
    tone: str = Form("professional"),
    format: Literal[
        "listicle", "how-to", "comparison", "case study", "myth-busting", "interview prep"
    ] = Form(...),
    feedback: str = Form("", max_length=1000),
):
    _check_rate_limit(request)
    _validate_fields(niche, audience, tone)
    try:
        result = await regenerate_one(niche.strip(), audience.strip(), tone, format, feedback)
    except (ProviderError, httpx.HTTPError, KeyError) as exc:
        logger.exception("Idea regeneration failed")
        raise HTTPException(status_code=502, detail=f"Idea regeneration failed: {exc}") from exc
    record_run(
        niche=niche.strip(),
        audience=audience.strip(),
        tone=tone,
        mode=result.mode,
        model=result.model,
        latency_ms=result.latency_ms,
        ideas=result.ideas,
        duplicate_rate=result.duplicate_rate,
    )
    return _response(result)

@app.get("/metrics")
async def get_metrics():
    return metrics_summary()


@app.get("/history.csv")
async def export_history(x_history_token: str | None = Header(None)):
    expected_token = os.getenv("HISTORY_EXPORT_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=404, detail="History export is disabled.")
    if not x_history_token or not secrets.compare_digest(x_history_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid history export token.")
    rows = metrics_summary(include_history=True)["recent_runs"]
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "created_at", "niche", "audience", "tone", "mode", "model", "latency_ms",
            "validation_pass_rate", "duplicate_rate", "validation_errors", "total", "ideas_json",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="generation-history.csv"'},
    )


@app.get("/models")
async def get_models():
    try:
        status = provider_status()
        return {"models": [status["model"]], "mode": status["mode"]}
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
