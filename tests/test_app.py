import sqlite3

from fastapi.testclient import TestClient
from fastapi import HTTPException
from types import SimpleNamespace

import main
from idea_engine import GenerationResult
from run_history import claim_daily_run, daily_runs, metrics_summary, record_vote


def test_homepage_displays_provider_and_brief_fields(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    with TestClient(main.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "freshers in India" in response.text
    assert "modeBadge" in response.text
    assert "Export Markdown" in response.text


def test_generate_and_metrics_are_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.sqlite3"))
    with TestClient(main.app) as client:
        response = client.post(
            "/generate",
            data={
                "niche": "Power BI",
                "audience": "freshers in India",
                "num_ideas": "3",
                "tone": "professional",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["ideas"]) == 3
        assert body["mode"] == "demo"
        assert body["model"] == "built-in samples"
        assert body["sources"]
        regenerated = client.post(
            "/regenerate",
            data={
                "niche": "Power BI",
                "audience": "freshers in India",
                "tone": "professional",
                "format": "comparison",
                "feedback": "Use a fresh comparison angle.",
            },
        )
        assert regenerated.status_code == 200, regenerated.text
        assert len(regenerated.json()["ideas"]) == 1
        metrics = client.get("/metrics").json()
        assert metrics["runs"] == 2
        assert metrics["ideas"] == 4
        assert metrics["validation_pass_rate"] == 1
        assert metrics["by_model_and_tone"][0]["model"] == "built-in samples"


def test_input_limits_are_enforced(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    with TestClient(main.app) as client:
        response = client.post("/generate", data={"niche": "x" * 121})
    assert response.status_code == 422


def test_requests_are_rate_limited_per_ip(monkeypatch):
    main.request_times.clear()
    monkeypatch.setattr(main, "REQUEST_LIMIT", 2)
    request = SimpleNamespace(client=SimpleNamespace(host="rate-limit-test"))
    main._check_rate_limit(request)
    main._check_rate_limit(request)
    try:
        main._check_rate_limit(request)
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected the per-IP request limit to reject a third request.")
    finally:
        main.request_times.clear()


def test_global_demo_daily_cap_is_atomic_and_reports_remaining(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "quota.sqlite3"))
    assert daily_runs() == 0
    assert claim_daily_run(2) == 1
    assert claim_daily_run(2) == 0
    assert claim_daily_run(2) is None
    assert daily_runs() == 2


def test_byok_header_is_passed_to_provider_not_saved(monkeypatch, tmp_path, caplog):
    api_key = "byok-test-secret"
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "byok.sqlite3"))
    monkeypatch.setattr(main, "DAILY_FREE_LIMIT", 0)
    captured = {}

    async def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return GenerationResult(
            ideas=[{"idea": {"title": "Example"}, "validation_errors": [], "valid": True}],
            sources=[],
            mode="openai",
            model="gpt-4o-mini",
            latency_ms=2.0,
            validation_pass_rate=1.0,
            duplicate_rate=0.0,
        )

    monkeypatch.setattr(main, "generate_ideas", fake_generate)
    with TestClient(main.app) as client:
        response = client.post(
            "/generate",
            data={"niche": "SQL", "audience": "students", "provider": "openai"},
            headers={"X-Provider-Key": api_key},
        )
    assert response.status_code == 200
    assert captured["api_key"] == api_key
    assert api_key not in caplog.text
    assert daily_runs() == 0
    content = (tmp_path / "byok.sqlite3").read_bytes()
    assert api_key.encode() not in content
    assert b"Example" not in content


def test_optional_writer_constraints_are_passed_to_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "constraints.sqlite3"))
    captured = {}

    async def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return GenerationResult([], [], "demo", "fixture", 1, 1, 0)

    monkeypatch.setattr(main, "generate_ideas", fake_generate)
    with TestClient(main.app) as client:
        response = client.post(
            "/generate",
            data={
                "niche": "SQL",
                "audience": "students in India",
                "provider": "demo",
                "target_keyword": "SQL interview questions",
                "est_word_count": "1500",
                "extra_notes": "Use beginner-friendly examples.",
            },
        )
    assert response.status_code == 200
    assert captured["target_keyword"] == "SQL interview questions"
    assert captured["est_word_count"] == 1500
    assert captured["extra_notes"] == "Use beginner-friendly examples."


def test_demo_daily_cap_rejects_more_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "full.sqlite3"))
    monkeypatch.setattr(main, "DAILY_FREE_LIMIT", 1)
    main.request_times.clear()
    with TestClient(main.app) as client:
        first = client.post("/generate", data={"niche": "SQL"})
        second = client.post("/generate", data={"niche": "SQL"})
    assert first.status_code == 200
    assert first.json()["remaining_daily_runs"] == 0
    assert second.status_code == 429
    assert "try again tomorrow" in second.json()["detail"]


def test_feedback_stores_anonymous_aggregate_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "votes.sqlite3"))
    with TestClient(main.app) as client:
        assert client.post("/feedback", data={"vote": "up"}).json() == {"ok": True}
        assert client.post("/feedback", data={"vote": "down"}).json() == {"ok": True}
        metrics = client.get("/metrics").json()
    assert metrics["votes"] == {"up": 1, "down": 1}


def test_legacy_run_history_is_reduced_to_aggregate_metrics(tmp_path, monkeypatch):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE generation_runs (
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, niche TEXT NOT NULL,
                audience TEXT NOT NULL, tone TEXT NOT NULL, mode TEXT NOT NULL,
                model TEXT NOT NULL, latency_ms REAL NOT NULL, passed INTEGER NOT NULL,
                total INTEGER NOT NULL, duplicate_rate REAL NOT NULL,
                validation_errors TEXT NOT NULL, ideas_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO generation_runs VALUES
            (1, '2025-01-02T12:00:00+00:00', 'private niche', 'private audience',
             'professional', 'demo', 'fixture', 12, 1, 1, 0, '{}', 'private idea')
            """
        )
    monkeypatch.setenv("DATABASE_PATH", str(database))
    metrics = metrics_summary()
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert metrics["runs"] == 1
    assert "generation_runs" not in tables
    stored_text = database.read_bytes()
    assert b"private niche" not in stored_text
    assert b"private idea" not in stored_text
