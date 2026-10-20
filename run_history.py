import json
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _connect() -> sqlite3.Connection:
    db_path = Path(os.getenv("DATABASE_PATH", "data/history.sqlite3"))
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_runs (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            niche TEXT NOT NULL,
            audience TEXT NOT NULL,
            tone TEXT NOT NULL,
            mode TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            latency_ms REAL NOT NULL,
            passed INTEGER NOT NULL,
            total INTEGER NOT NULL,
            duplicate_rate REAL NOT NULL,
            validation_errors TEXT NOT NULL,
            ideas_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(generation_runs)")
    }
    if "ideas_json" not in columns:
        connection.execute(
            "ALTER TABLE generation_runs ADD COLUMN ideas_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "model" not in columns:
        connection.execute(
            "ALTER TABLE generation_runs ADD COLUMN model TEXT NOT NULL DEFAULT ''"
        )
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


def record_run(
    niche: str,
    audience: str,
    tone: str,
    mode: str,
    model: str,
    latency_ms: float,
    ideas: list[dict],
    duplicate_rate: float,
) -> None:
    errors = Counter(
        error
        for result in ideas
        for error in result.get("validation_errors", [])
    )
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO generation_runs
            (created_at, niche, audience, tone, mode, model, latency_ms, passed, total,
             duplicate_rate, validation_errors, ideas_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                niche,
                audience,
                tone,
                mode,
                model,
                latency_ms,
                sum(not idea.get("validation_errors") for idea in ideas),
                len(ideas),
                duplicate_rate,
                json.dumps(errors),
                json.dumps(ideas),
            ),
        )


def metrics_summary(include_history: bool = False) -> dict:
    with _connection() as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS runs,
                   COALESCE(SUM(passed), 0) AS passed,
                   COALESCE(SUM(total), 0) AS ideas,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(AVG(duplicate_rate), 0) AS avg_duplicate_rate
            FROM generation_runs
            """
        ).fetchone()
        dimensions = connection.execute(
            """
            SELECT mode, model, tone, COUNT(*) AS runs,
                   COALESCE(SUM(passed), 0) AS passed,
                   COALESCE(SUM(total), 0) AS ideas,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(AVG(duplicate_rate), 0) AS avg_duplicate_rate
            FROM generation_runs
            GROUP BY mode, model, tone
            ORDER BY runs DESC, mode, model, tone
            """
        ).fetchall()
        history_query = """
            SELECT created_at, niche, audience, tone, mode, model, latency_ms,
                   passed, total, duplicate_rate, validation_errors, ideas_json
            FROM generation_runs
            ORDER BY id DESC
        """
        if not include_history:
            history_query += " LIMIT 50"
        rows = connection.execute(
            history_query
        ).fetchall()
    failure_counts: Counter[str] = Counter()
    recent_runs = []
    for row in rows:
        run = dict(row)
        errors = json.loads(run["validation_errors"])
        failure_counts.update(errors)
        run["validation_errors"] = json.dumps(errors)
        run["validation_pass_rate"] = run.pop("passed") / run["total"] if run["total"] else 0
        recent_runs.append(run)
    return {
        "runs": totals["runs"],
        "ideas": totals["ideas"],
        "validation_pass_rate": totals["passed"] / totals["ideas"] if totals["ideas"] else 0,
        "avg_latency_ms": round(totals["avg_latency_ms"], 2),
        "avg_duplicate_rate": round(totals["avg_duplicate_rate"], 3),
        "by_model_and_tone": [
            {
                "mode": row["mode"],
                "model": row["model"],
                "tone": row["tone"],
                "runs": row["runs"],
                "validation_pass_rate": row["passed"] / row["ideas"] if row["ideas"] else 0,
                "avg_latency_ms": round(row["avg_latency_ms"], 2),
                "avg_duplicate_rate": round(row["avg_duplicate_rate"], 3),
            }
            for row in dimensions
        ],
        "failure_types": dict(failure_counts.most_common()),
        "recent_runs": recent_runs if include_history else [],
    }
