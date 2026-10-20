import json
from pathlib import Path

import pytest

import idea_engine
from idea_engine import (
    BlogIdea,
    FORMATS,
    Provider,
    _parse_ideas,
    _prompt,
    _generate_batch,
    demo_ideas_from_prompt,
    generate_ideas,
    get_provider,
    lexical_similarity,
    _duplicate_pairs,
    validate_idea,
)
from knowledge_base import retrieve_sources


NICHE = "Power BI"
AUDIENCE = "freshers in India"


def valid_idea(**updates):
    idea = {
        "title": "Power BI for Freshers in India: Five Skills to Practice",
        "hook": "A practical learning path for freshers in India exploring Power BI.",
        "angle": "Build confidence by practicing five specific Power BI skills.",
        "target_keyword": "Power BI for freshers",
        "search_intent": "informational",
        "audience": AUDIENCE,
        "est_word_count": 1200,
        "format": "listicle",
        "outline": [
            {"heading": "Introduction", "summary": "Explain why these Power BI skills matter to freshers in India."},
            *[
                {
                    "heading": f"Skill {number}: {topic}",
                    "summary": f"Practice {topic} in a Power BI task designed for freshers in India.",
                }
                for number, topic in enumerate(
                    ("data modeling", "visual design", "filters", "storytelling", "portfolio work"), 1
                )
            ],
            {"heading": "Conclusion and next steps", "summary": "Choose one Power BI skill and practice it this week."},
        ],
    }
    idea.update(updates)
    return BlogIdea.model_validate(idea)


def test_valid_idea_passes_all_checks():
    idea = valid_idea()
    assert 50 <= len(idea.title) <= 70
    assert validate_idea(idea, NICHE, AUDIENCE) == []


@pytest.mark.parametrize("title", [
    "Too short",
    "Power BI for Freshers in India: A Detailed and Extremely Long Headline That Goes Beyond the Limit",
])
def test_title_length_is_validated(title):
    assert any("50-70 characters" in issue for issue in validate_idea(valid_idea(title=title), NICHE, AUDIENCE))


def test_outline_requires_intro_three_to_six_sections_and_conclusion():
    idea = valid_idea(outline=[
        {"heading": "Overview", "summary": "A short section that does not identify an introduction."},
        {"heading": "One topic", "summary": "A useful detailed section about Power BI for freshers in India."},
        {"heading": "Conclusion", "summary": "A clear summary with a next step for the reader."},
    ])
    errors = validate_idea(idea, NICHE, AUDIENCE)
    assert any("introduction" in issue.lower() for issue in errors)
    assert any("3-6 substantive" in issue for issue in errors)


def test_number_in_title_must_be_covered_by_outline():
    idea = valid_idea(outline=valid_idea().outline[:-2] + valid_idea().outline[-1:])
    assert any("does not enumerate all 5" in issue for issue in validate_idea(idea, NICHE, AUDIENCE))


def test_number_before_niche_name_is_still_checked():
    idea = valid_idea(
        title="5 Power BI Skills Every Fresher in India Should Practice",
        outline=valid_idea().outline[:-2] + valid_idea().outline[-1:],
    )
    assert any("does not enumerate all 5" in issue for issue in validate_idea(idea, NICHE, AUDIENCE))


def test_idea_must_match_requested_niche_and_audience():
    idea = valid_idea(
        title="Content Marketing for Freshers in India: Five Skills to Practice",
        hook="A practical learning path for freshers in India exploring content marketing.",
        angle="Practice five writing skills.",
        target_keyword="content marketing for freshers",
        outline=[
            {"heading": "Introduction", "summary": "Explain why these writing skills matter to freshers in India."},
            *[
                {"heading": f"Skill {number}", "summary": f"Practice writing topic {number} for freshers in India."}
                for number in range(1, 6)
            ],
            {"heading": "Conclusion", "summary": "Choose a skill to practice next."},
        ],
    )
    errors = validate_idea(idea, NICHE, AUDIENCE)
    assert any("niche" in issue for issue in errors)

    wrong_audience = valid_idea(audience="experienced managers")
    assert any("audience field" in issue.lower() for issue in validate_idea(wrong_audience, NICHE, AUDIENCE))


def test_unsupported_claims_and_unverifiable_statistics_are_flagged():
    idea = valid_idea(angle="Power BI is revolutionizing the industry and increases productivity by 25%.")
    errors = validate_idea(idea, NICHE, AUDIENCE)
    assert any("Unsupported promotional claim" in issue for issue in errors)
    assert any("Unverifiable statistic" in issue for issue in errors)


def test_near_duplicate_ideas_have_high_similarity():
    original = valid_idea()
    near_duplicate = valid_idea(
        title="Power BI for Freshers in India: Five Skills to Practice",
        angle="Practice five specific Power BI skills with a practical learning path.",
    )
    assert lexical_similarity(original, near_duplicate) > 0.84


def test_duplicate_target_keywords_are_detected_even_with_different_titles():
    first = valid_idea()
    second = valid_idea(
        title="Power BI Portfolio Projects for Freshers in India",
        target_keyword="POWER-BI for freshers",
        angle="Build one small portfolio dashboard and explain its design choices.",
    )
    assert _duplicate_pairs([first, second], embeddings=None) == [(1, 0)]


def test_knowledge_retrieval_returns_citations():
    sources = retrieve_sources("Power BI freshers")
    assert sources
    assert all({"id", "title", "content"} <= source.keys() for source in sources)
    assert any("Power BI" in source["title"] for source in sources)


def test_unknown_niche_uses_labelled_generic_fallback():
    sources = retrieve_sources("underwater basket weaving")
    assert sources
    assert all(source["title"].startswith("Generic fallback:") for source in sources)


def test_tone_changes_demo_output():
    professional_prompt = _prompt(NICHE, AUDIENCE, "professional", ["listicle"], [])
    conversational_prompt = _prompt(NICHE, AUDIENCE, "conversational", ["listicle"], [])
    professional = demo_ideas_from_prompt(professional_prompt)[0]
    conversational = demo_ideas_from_prompt(conversational_prompt)[0]
    assert professional["hook"] != conversational["hook"]


def test_auto_mode_uses_available_small_ollama_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(
        idea_engine,
        "_available_ollama_models",
        lambda: ["llama3.1:8b", "gemma2:2b"],
    )
    provider = get_provider()
    assert provider.mode == "ollama"
    assert provider.model == "gemma2:2b"


def test_auto_mode_uses_demo_when_no_provider_is_available(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(idea_engine, "_available_ollama_models", lambda: [])
    assert get_provider().mode == "demo"


def test_default_provider_is_demo_without_configured_credentials(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_provider().mode == "demo"


def test_bring_your_own_provider_key_is_held_only_by_provider_instance():
    provider = get_provider("openai", "test-key-never-persist")
    assert provider.mode == "openai"
    assert provider.api_key == "test-key-never-persist"


@pytest.mark.asyncio
async def test_openai_byok_key_is_sent_only_to_provider_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"ideas": []}'}}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, headers, json):
            captured.update(headers=headers, body=json)
            return FakeResponse()

    monkeypatch.setattr(idea_engine.httpx, "AsyncClient", FakeClient)
    key = "short-lived-test-key"
    await Provider("openai", key).generate("sample prompt")
    assert captured["headers"]["Authorization"].endswith(key)
    assert "short-lived-test-key" not in captured["body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_demo_can_load_a_recorded_sample_with_explicit_label(monkeypatch, tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "llama31_samples.json").write_text(
        json.dumps(
            [{
                "niche": NICHE,
                "audience": "students and freshers in India",
                "ideas": [{"valid": True, "idea": valid_idea(
                    title="Power BI for Freshers in India: Five Skills to Practice",
                    audience="students and freshers in India",
                ).model_dump()}],
            }]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(idea_engine, "__file__", str(tmp_path / "idea_engine.py"))
    prompt = _prompt(
        NICHE, "freshers in India", "professional", ["listicle"], [], est_word_count=1200
    )
    provider = Provider("demo")
    output = json.loads(await provider.generate(prompt))
    assert provider.model.startswith("recorded from llama3.1:8b")
    assert output["ideas"][0]["title"].startswith("Power BI for Freshers")


def test_recorded_llama_fixtures_cover_eight_niches():
    path = Path(idea_engine.__file__).resolve().parent / "fixtures" / "llama31_samples.json"
    recordings = json.loads(path.read_text(encoding="utf-8"))
    assert len(recordings) == 8
    assert len({item["niche"] for item in recordings}) == 8
    assert all(item["model"] == "llama3.1:8b" for item in recordings)
    assert all(item["source"].startswith("Recorded from llama3.1:8b") for item in recordings)
    assert all(
        idea_engine._recorded_demo_ideas(
            _prompt(
                item["niche"],
                "freshers in India",
                "professional",
                ["listicle"],
                [],
                est_word_count=1200,
            )
        )
        for item in recordings
    )


def test_each_format_has_a_distinct_demo_outline():
    ideas = demo_ideas_from_prompt(_prompt(NICHE, AUDIENCE, "professional", list(FORMATS), []))
    assert len({idea["format"] for idea in ideas}) == len(FORMATS)
    assert len({tuple(section["heading"] for section in idea["outline"]) for idea in ideas}) == len(FORMATS)


def test_structured_output_parsing_rejects_invalid_json_and_shape():
    with pytest.raises(json.JSONDecodeError):
        _parse_ideas("{not json", ["listicle"])
    with pytest.raises(ValueError, match="Expected 1 ideas"):
        _parse_ideas('{"ideas": []}', ["listicle"])


@pytest.mark.asyncio
async def test_demo_generation_returns_distinct_valid_structured_ideas(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    result = await generate_ideas(NICHE, AUDIENCE, 3, "professional")
    assert len(result.ideas) == 3
    assert len({item["idea"]["format"] for item in result.ideas}) == 3
    assert all(not item["validation_errors"] for item in result.ideas)
    assert len({item["idea"]["target_keyword"].casefold() for item in result.ideas}) == 3
    assert result.duplicate_rate == 0
    assert set(result.step_timings) == {"prompt_ms", "provider_ms", "validation_ms"}
    assert all(value >= 0 for value in result.step_timings.values())


@pytest.mark.asyncio
async def test_invalid_json_is_retried():
    idea = valid_idea().model_dump()

    class FakeProvider:
        def __init__(self):
            self.responses = ["not json", json.dumps({"ideas": [idea]})]

        async def generate(self, prompt):
            return self.responses.pop(0)

    batch = await _generate_batch(FakeProvider(), "prompt", ["listicle"])
    assert len(batch) == 1
    assert batch[0].title == idea["title"]


@pytest.mark.asyncio
async def test_reflection_regenerates_only_the_failing_idea(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    invalid = valid_idea(title="Too short")
    improved = valid_idea()
    calls = []

    async def fake_batch(_provider, prompt, formats):
        calls.append((prompt, formats))
        return [invalid if len(calls) == 1 else improved]

    monkeypatch.setattr(idea_engine, "_generate_batch", fake_batch)
    result = await generate_ideas(NICHE, AUDIENCE, 1, "professional")
    assert len(calls) == 2
    assert calls[1][1] == ["listicle"]
    assert "50-70 characters" in calls[1][0]
    assert result.ideas[0]["valid"]


@pytest.mark.asyncio
async def test_duplicate_keyword_reflection_retries_only_duplicate_idea(monkeypatch):
    first = valid_idea()
    duplicate = valid_idea(
        title="Power BI Portfolio Projects for Freshers in India Today",
        target_keyword=first.target_keyword,
        format="how-to",
    )
    improved = valid_idea(
        title="How Freshers in India Build a Power BI Portfolio Dashboard",
        hook="A first dashboard exercise using a small dataset to explain a practical decision.",
        angle="Design one visual, check its limitations, and narrate the choices behind a Power BI dashboard.",
        target_keyword="Power BI beginner dashboard portfolio",
        format="how-to",
    )
    calls = []

    async def fake_batch(_provider, prompt, formats):
        calls.append((prompt, formats))
        return [first, duplicate] if len(calls) == 1 else [improved]

    monkeypatch.setattr(idea_engine, "_generate_batch", fake_batch)
    result = await generate_ideas(NICHE, AUDIENCE, 2, "professional", provider_name="demo")
    assert len(calls) > 1
    assert all(formats == ["how-to"] for _, formats in calls[1:])
    assert "too similar" in calls[1][0]
    assert result.duplicate_rate == 0
