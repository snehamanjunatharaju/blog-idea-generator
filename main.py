import json
import logging
import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from idea_engine import ProviderError, generate_ideas, get_provider, provider_status, regenerate_one
from run_history import claim_daily_run, daily_runs, metrics_summary, record_run, record_vote

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("blog_idea_generator")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

REQUEST_LIMIT = int(os.getenv("REQUESTS_PER_MINUTE", "10"))
DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_RUNS", "500"))
request_times: dict[str, deque[float]] = defaultdict(deque)
request_lock = Lock()
TONES = {"professional", "conversational", "humorous", "inspirational", "educational"}
ABUSIVE_PROMPT_TERMS = ("kill yourself", "racial slur", "sexual violence")


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
    if any(term in f"{niche} {audience}".lower() for term in ABUSIVE_PROMPT_TERMS):
        raise HTTPException(status_code=422, detail="Please use a respectful niche and audience.")


def _response(result) -> dict:
    return {
        "ideas": result.ideas,
        "sources": result.sources,
        "mode": result.mode,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "validation_pass_rate": result.validation_pass_rate,
        "duplicate_rate": result.duplicate_rate,
        "remaining_daily_runs": max(0, DAILY_FREE_LIMIT - daily_runs()),
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
        status = provider_status()
        status["remaining_daily_runs"] = max(0, DAILY_FREE_LIMIT - daily_runs())
        status["daily_free_limit"] = DAILY_FREE_LIMIT
        return status
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
    target_keyword: str = Form("", max_length=100),
    est_word_count: int = Form(1200, ge=300, le=5000),
    extra_notes: str = Form("", max_length=1000),
    provider: Literal["demo", "ollama", "anthropic", "openai"] = Form("demo"),
    api_key: str | None = Header(None, alias="X-Provider-Key"),
):
    _check_rate_limit(request)
    _validate_fields(niche, audience, tone)
    started = time.perf_counter()
    try:
        if api_key and (len(api_key) > 300 or "\n" in api_key or "\r" in api_key):
            raise HTTPException(status_code=422, detail="The provider key format is invalid.")
        selected_provider = get_provider(provider, api_key)
        remaining = None
        if selected_provider.mode == "demo":
            remaining = claim_daily_run(DAILY_FREE_LIMIT)
            if remaining is None:
                raise HTTPException(
                    status_code=429,
                    detail="Today’s free demo runs are used up. Please try again tomorrow or use your own provider key.",
                )
        result = await generate_ideas(
            niche.strip(), audience.strip(), num_ideas, tone, model or None,
            provider_name=provider, api_key=api_key,
            target_keyword=target_keyword.strip(),
            est_word_count=est_word_count,
            extra_notes=extra_notes.strip(),
        )
    except (ProviderError, httpx.HTTPError, KeyError) as exc:
        logger.error("Idea generation failed provider=%s error_type=%s", provider, type(exc).__name__)
        if isinstance(exc, ProviderError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail="The provider could not complete this request. Please retry.") from exc
    except (ValidationError, ValueError) as exc:
        logger.exception("Provider returned invalid idea data")
        raise HTTPException(status_code=502, detail="Provider returned invalid idea data.") from exc

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    record_run(
        mode=result.mode,
        model=result.model,
        tone=tone,
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
    provider: Literal["demo", "ollama", "anthropic", "openai"] = Form("demo"),
    api_key: str | None = Header(None, alias="X-Provider-Key"),
):
    _check_rate_limit(request)
    _validate_fields(niche, audience, tone)
    try:
        selected_provider = get_provider(provider, api_key)
        if selected_provider.mode == "demo":
            if claim_daily_run(DAILY_FREE_LIMIT) is None:
                raise HTTPException(
                    status_code=429,
                    detail="Today’s free demo runs are used up. Please try again tomorrow or use your own provider key.",
                )
        result = await regenerate_one(
            niche.strip(), audience.strip(), tone, format, feedback,
            provider_name=provider, api_key=api_key,
        )
    except (ProviderError, httpx.HTTPError, KeyError) as exc:
        logger.error("Idea regeneration failed provider=%s error_type=%s", provider, type(exc).__name__)
        if isinstance(exc, ProviderError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail="The provider could not complete this request. Please retry.") from exc
    record_run(
        mode=result.mode,
        model=result.model,
        tone=tone,
        latency_ms=result.latency_ms,
        ideas=result.ideas,
        duplicate_rate=result.duplicate_rate,
    )
    return _response(result)

@app.get("/metrics")
async def get_metrics():
    return metrics_summary()


@app.post("/feedback")
async def submit_feedback(vote: Literal["up", "down"] = Form(...)):
    record_vote(vote)
    return {"ok": True}


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
