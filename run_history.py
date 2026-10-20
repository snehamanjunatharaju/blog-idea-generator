import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _connect() -> sqlite3.Connection:
    raw_path = os.getenv("DATABASE_PATH", "data/history.sqlite3")
    if raw_path == ":memory:":
        connection = sqlite3.connect(raw_path, timeout=10)
    else:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_usage (
            day TEXT PRIMARY KEY,
            runs INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS run_metrics (
            id INTEGER PRIMARY KEY,
            day TEXT NOT NULL,
            mode TEXT NOT NULL,
            model TEXT NOT NULL,
            tone TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            passed INTEGER NOT NULL,
            total INTEGER NOT NULL,
            duplicate_rate REAL NOT NULL,
            prompt_ms REAL NOT NULL DEFAULT 0,
            provider_ms REAL NOT NULL DEFAULT 0,
            validation_ms REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS anonymous_votes (
            day TEXT NOT NULL,
            vote TEXT NOT NULL CHECK (vote IN ('up', 'down')),
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(day, vote)
        );
        """
    )
    legacy_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'generation_runs'"
    ).fetchone()
    if legacy_table:
        connection.execute("BEGIN IMMEDIATE")
        legacy_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'generation_runs'"
        ).fetchone()
        if legacy_table:
            connection.execute("PRAGMA secure_delete = ON")
            connection.execute(
                """
                INSERT INTO run_metrics
                (day, mode, model, tone, latency_ms, passed, total, duplicate_rate)
                SELECT substr(created_at, 1, 10), mode, model, tone, latency_ms,
                       passed, total, duplicate_rate
                FROM generation_runs
                """
            )
            connection.execute("DROP TABLE generation_runs")
    connection.commit()
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def daily_runs() -> int:
    with _connection() as connection:
        row = connection.execute(
            "SELECT runs FROM daily_usage WHERE day = ?", (_today(),)
        ).fetchone()
    return row["runs"] if row else 0


def claim_daily_run(limit: int) -> int | None:
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        day = _today()
        connection.execute(
            "INSERT OR IGNORE INTO daily_usage(day, runs) VALUES (?, 0)", (day,)
        )
        row = connection.execute(
            "SELECT runs FROM daily_usage WHERE day = ?", (day,)
        ).fetchone()
        if row["runs"] >= limit:
            return None
        remaining = limit - row["runs"] - 1
        connection.execute(
            "UPDATE daily_usage SET runs = runs + 1 WHERE day = ?", (day,)
        )
        return remaining


def record_run(
    mode: str,
    model: str,
    tone: str,
    latency_ms: float,
    ideas: list[dict],
    duplicate_rate: float,
    step_timings: dict[str, float] | None = None,
) -> None:
    timings = step_timings or {}
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO run_metrics
            (day, mode, model, tone, latency_ms, passed, total, duplicate_rate,
             prompt_ms, provider_ms, validation_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _today(),
                mode,
                model,
                tone,
                latency_ms,
                sum(not idea.get("validation_errors") for idea in ideas),
                len(ideas),
                duplicate_rate,
                timings.get("prompt_ms", 0),
                timings.get("provider_ms", 0),
                timings.get("validation_ms", 0),
            ),
        )


def record_vote(vote: str) -> None:
    if vote not in {"up", "down"}:
        raise ValueError("Vote must be up or down.")
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO anonymous_votes(day, vote, count) VALUES (?, ?, 1)
            ON CONFLICT(day, vote) DO UPDATE SET count = count + 1
            """,
            (_today(), vote),
        )


def metrics_summary() -> dict:
    with _connection() as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS runs, COALESCE(SUM(passed), 0) AS passed,
                   COALESCE(SUM(total), 0) AS ideas,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(AVG(duplicate_rate), 0) AS avg_duplicate_rate,
                   COALESCE(AVG(prompt_ms), 0) AS avg_prompt_ms,
                   COALESCE(AVG(provider_ms), 0) AS avg_provider_ms,
                   COALESCE(AVG(validation_ms), 0) AS avg_validation_ms
            FROM run_metrics
            """
        ).fetchone()
        dimensions = connection.execute(
            """
            SELECT mode, model, tone, COUNT(*) AS runs,
                   COALESCE(SUM(passed), 0) AS passed,
                   COALESCE(SUM(total), 0) AS ideas,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(AVG(duplicate_rate), 0) AS avg_duplicate_rate
            FROM run_metrics GROUP BY mode, model, tone
            ORDER BY runs DESC, mode, model, tone
            """
        ).fetchall()
        votes = connection.execute(
            "SELECT vote, COALESCE(SUM(count), 0) AS count FROM anonymous_votes GROUP BY vote"
        ).fetchall()
    vote_counts = {row["vote"]: row["count"] for row in votes}
    runs_today = daily_runs()
    return {
        "runs": totals["runs"],
        "runs_today": runs_today,
        "ideas": totals["ideas"],
        "validation_pass_rate": totals["passed"] / totals["ideas"] if totals["ideas"] else 0,
        "avg_latency_ms": round(totals["avg_latency_ms"], 2),
        "avg_duplicate_rate": round(totals["avg_duplicate_rate"], 3),
        "step_latency_ms": {
            "prompt": round(totals["avg_prompt_ms"], 2),
            "provider": round(totals["avg_provider_ms"], 2),
            "validation": round(totals["avg_validation_ms"], 2),
        },
        "by_model_and_tone": [
            {
                "mode": row["mode"], "model": row["model"], "tone": row["tone"],
                "runs": row["runs"],
                "validation_pass_rate": row["passed"] / row["ideas"] if row["ideas"] else 0,
                "avg_latency_ms": round(row["avg_latency_ms"], 2),
                "avg_duplicate_rate": round(row["avg_duplicate_rate"], 3),
            }
            for row in dimensions
        ],
        "votes": {"up": vote_counts.get("up", 0), "down": vote_counts.get("down", 0)},
        "data_stored": "anonymous aggregate metrics only; no prompts, ideas, API keys, or IP addresses",
    }
