from fastapi.testclient import TestClient
from fastapi import HTTPException
from types import SimpleNamespace

import main


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
    monkeypatch.setenv("HISTORY_EXPORT_TOKEN", "test-export-token")
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
        assert metrics["recent_runs"] == []
        assert metrics["by_model_and_tone"][0]["model"] == "built-in samples"
        assert client.get("/history.csv").status_code == 403
        csv_response = client.get(
            "/history.csv",
            headers={"X-History-Token": "test-export-token"},
        )
        assert csv_response.status_code == 200
        assert "Power BI" in csv_response.text


def test_history_export_is_disabled_without_a_token(monkeypatch):
    monkeypatch.delenv("HISTORY_EXPORT_TOKEN", raising=False)
    with TestClient(main.app) as client:
        response = client.get("/history.csv")
    assert response.status_code == 404


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
