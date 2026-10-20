import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from knowledge_base import retrieve_sources

logger = logging.getLogger(__name__)

FORMATS = (
    "listicle",
    "how-to",
    "comparison",
    "case study",
    "myth-busting",
    "interview prep",
)
MAX_JSON_ATTEMPTS = 2
MAX_REFLECTION_ATTEMPTS = 2
DUPLICATE_THRESHOLD = 0.84
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(revolutioniz(?:e|es|ing|ed)|must-have|guaranteed to|game[- ]changing|"
    r"industry-leading|proven to|skyrocket(?:s|ed)?)\b",
    re.IGNORECASE,
)
UNSUPPORTED_STAT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x\b|times\b)", re.I)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
NUMBERED_PROMISE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+"
    r"(?:(?:[\w-]+)\s+){0,2}?"
    r"(?:ways|tips|steps|questions|mistakes|skills|strategies|reasons|"
    r"tools|examples|lessons|myths|checks|ideas)\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "with",
    "your", "you", "from", "by", "into", "as", "at",
}


class OutlineSection(BaseModel):
    heading: str = Field(min_length=3, max_length=100)
    summary: str = Field(min_length=10, max_length=500)


class BlogIdea(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    hook: str = Field(min_length=1, max_length=500)
    angle: str = Field(min_length=1, max_length=500)
    target_keyword: str = Field(min_length=1, max_length=100)
    search_intent: Literal["informational", "commercial", "transactional", "navigational"]
    audience: str = Field(min_length=1, max_length=120)
    est_word_count: int = Field(ge=300, le=5000)
    format: Literal[
        "listicle", "how-to", "comparison", "case study", "myth-busting", "interview prep"
    ]
    outline: list[OutlineSection] = Field(min_length=1, max_length=10)


class IdeaBatch(BaseModel):
    ideas: list[BlogIdea]


@dataclass
class GenerationResult:
    ideas: list[dict]
    sources: list[dict]
    mode: str
    model: str
    latency_ms: float
    validation_pass_rate: float
    duplicate_rate: float


class ProviderError(RuntimeError):
    pass


class Provider:
    def __init__(self, mode: str):
        self.mode = mode
        self.model = os.getenv("LLM_MODEL", "")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    async def generate(self, prompt: str) -> str:
        if self.mode == "demo":
            return json.dumps({"ideas": demo_ideas_from_prompt(prompt)})
        if self.mode == "ollama":
            model = self.model or "llama3.2"
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                )
                response.raise_for_status()
                return response.json()["response"]
        if self.mode == "openai":
            key = os.environ["OPENAI_API_KEY"]
            model = self.model or "gpt-4o-mini"
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        if self.mode == "anthropic":
            key = os.environ["ANTHROPIC_API_KEY"]
            model = self.model or "claude-3-5-haiku-latest"
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
        raise ProviderError(f"Unsupported provider: {self.mode}")

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.mode != "ollama":
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), "input": texts},
                )
                response.raise_for_status()
                vectors = response.json().get("embeddings")
                if vectors and len(vectors) == len(texts):
                    return vectors
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            logger.info("Ollama embeddings unavailable; using local lexical similarity")
        return None


def get_provider() -> Provider:
    requested = os.getenv("LLM_PROVIDER", "auto").lower()
    if requested == "demo":
        return Provider("demo")
    if requested in {"ollama", "openai", "anthropic"}:
        if requested == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is required for the OpenAI provider.")
        if requested == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is required for the Anthropic provider.")
        provider = Provider(requested)
        if requested == "ollama" and not provider.model:
            provider.model = _default_ollama_model() or "llama3.2"
        if requested == "ollama":
            available = _available_ollama_models()
            if available and provider.model not in available:
                raise ProviderError(
                    f"Ollama model '{provider.model}' is not installed. Set LLM_MODEL to an installed model."
                )
        return provider
    if requested != "auto":
        raise ProviderError("LLM_PROVIDER must be auto, demo, ollama, openai, or anthropic.")
    if os.getenv("ANTHROPIC_API_KEY"):
        return Provider("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        return Provider("openai")
    available = _available_ollama_models()
    if available:
        provider = Provider("ollama")
        provider.model = _default_ollama_model(available)
        return provider
    return Provider("demo")


def _available_ollama_models() -> list[str]:
    try:
        with httpx.Client(timeout=0.3) as client:
            response = client.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/tags"
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models if model.get("name")]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return []


def _default_ollama_model(models: list[str] | None = None) -> str:
    models = _available_ollama_models() if models is None else models
    configured = os.getenv("LLM_MODEL")
    if configured:
        if models and configured not in models:
            raise ProviderError(
                f"Ollama model '{configured}' is not installed. Set LLM_MODEL to an installed model."
            )
        return configured
    if "gemma2:2b" in models:
        return "gemma2:2b"
    return models[0] if models else ""


def provider_status() -> dict:
    provider = get_provider()
    return {"mode": provider.mode, "model": provider.model or _default_model(provider.mode)}


def _default_model(mode: str) -> str:
    return {
        "demo": "built-in samples",
        "ollama": "llama3.2",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
    }[mode]


def _prompt(
    niche: str,
    audience: str,
    tone: str,
    formats: list[str],
    sources: list[dict],
    feedback: str = "",
) -> str:
    source_text = "\n".join(f"- {source['title']}: {source['content']}" for source in sources)
    tone_text = {
        "professional": "Use precise, confident language and avoid slang.",
        "conversational": "Write warmly and plainly, as a helpful peer; use contractions naturally.",
        "humorous": "Use light, relevant wit without sacrificing clarity or making claims.",
        "inspirational": "Emphasize achievable progress without hype or guarantees.",
        "educational": "Prioritize clear explanations, definitions, and practical learning outcomes.",
    }[tone]
    return f"""
Create exactly {len(formats)} distinct blog ideas for the niche "{niche}" and audience "{audience}".
Use each assigned format once, in this order: {json.dumps(formats)}.
Tone: {tone_text}

Each idea must be one JSON object with title, hook, angle, target_keyword, search_intent
(informational, commercial, transactional, or navigational), audience, est_word_count,
format, and outline. Each outline is an array of objects with heading and summary.
Titles must contain 50-70 characters. Outlines must have an introduction, 3-6 substantive
sections, and a conclusion. Keep every idea specific to the niche and audience. If a title
promises a number of tips, steps, questions, skills, etc., provide exactly that many
corresponding items in the substantive outline sections. Avoid unsupported claims and
statistics. Do not invent research, numbers, sources, or guarantees.
Use these notes as grounding, not as facts about the niche:
{source_text}
{f'Correct only this idea. Prior validation feedback: {feedback}' if feedback else ''}
Return only a JSON object of the form {{"ideas": [...]}} with no markdown fences.
""".strip()


def _parse_ideas(text: str, expected_formats: list[str]) -> list[BlogIdea]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    data = json.loads(cleaned)
    if isinstance(data, list):
        data = {"ideas": data}
    batch = IdeaBatch.model_validate(data)
    if len(batch.ideas) != len(expected_formats):
        raise ValueError(f"Expected {len(expected_formats)} ideas; received {len(batch.ideas)}.")
    actual_formats = [idea.format for idea in batch.ideas]
    if actual_formats != expected_formats:
        raise ValueError(f"Expected formats {expected_formats}; received {actual_formats}.")
    return batch.ideas


async def _generate_batch(provider: Provider, prompt: str, formats: list[str]) -> list[BlogIdea]:
    feedback = ""
    for attempt in range(MAX_JSON_ATTEMPTS):
        raw = await provider.generate(prompt if not feedback else f"{prompt}\n{feedback}")
        try:
            return _parse_ideas(raw, formats)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.info("Structured output attempt %d failed: %s", attempt + 1, exc)
            feedback = f"Your previous output was invalid: {exc}. Return corrected valid JSON."
    raise ProviderError("The model did not return valid structured ideas after retrying.")


def validate_idea(idea: BlogIdea, niche: str, audience: str) -> list[str]:
    errors: list[str] = []
    title_len = len(idea.title.strip())
    if not 50 <= title_len <= 70:
        errors.append(f"Title must be 50-70 characters (currently {title_len}).")

    headings = [section.heading.strip().lower() for section in idea.outline]
    intro = any(re.search(r"\b(introduction|intro|getting started)\b", heading) for heading in headings)
    conclusion = any(re.search(r"\b(conclusion|wrap[- ]?up|next steps|key takeaways)\b", heading) for heading in headings)
    body_count = len(idea.outline) - int(intro) - int(conclusion)
    if not intro:
        errors.append("Outline needs an introduction.")
    if not conclusion:
        errors.append("Outline needs a conclusion.")
    if not 3 <= body_count <= 6:
        errors.append(f"Outline needs 3-6 substantive sections (currently {body_count}).")

    promised = NUMBERED_PROMISE.search(idea.title)
    if promised:
        count_text = promised.group(1).lower()
        count = int(count_text) if count_text.isdigit() else NUMBER_WORDS[count_text]
        expected = {str(number) for number in range(1, count + 1)}
        found = set(
            re.findall(
                r"\b(?:item|skill|step|tip|question|mistake|strategy|reason|tool|example)"
                r"\s*(?:#|:)?\s*(\d{1,2})\b",
                " ".join(section.heading for section in idea.outline),
                re.IGNORECASE,
            )
        )
        if count > 20 or not expected.issubset(found):
            errors.append(
                f"Title promises {count} items but outline does not enumerate all {count}."
            )

    content = " ".join(
        [idea.title, idea.hook, idea.angle, idea.target_keyword]
        + [f"{section.heading} {section.summary}" for section in idea.outline]
    ).lower()
    niche_terms = [term for term in _terms(niche) if term not in STOP_WORDS]
    content_terms = _terms(content)
    if niche_terms and not set(niche_terms).issubset(content_terms):
        errors.append("Idea does not clearly match the requested niche.")
    audience_terms = [term for term in _terms(audience) if term not in STOP_WORDS]
    idea_audience_terms = _terms(idea.audience)
    if audience_terms and not set(audience_terms).issubset(idea_audience_terms):
        errors.append("Audience field does not match the requested audience.")
    if audience_terms and not set(audience_terms).issubset(content_terms):
        errors.append("Idea does not clearly address the requested audience.")

    full_text = " ".join(
        [idea.title, idea.hook, idea.angle]
        + [f"{section.heading} {section.summary}" for section in idea.outline]
    )
    if unsupported := UNSUPPORTED_CLAIMS.search(full_text):
        errors.append(f"Unsupported promotional claim: '{unsupported.group(0)}'.")
    if stat := UNSUPPORTED_STAT.search(full_text):
        errors.append(f"Unverifiable statistic: '{stat.group(0)}'.")
    return errors


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _idea_score(errors: list[str]) -> int:
    return -len(errors)


async def generate_ideas(
    niche: str,
    audience: str,
    num_ideas: int,
    tone: str,
    model: str | None = None,
) -> GenerationResult:
    started = time.perf_counter()
    provider = get_provider()
    if model:
        provider.model = model
    formats = list(FORMATS[:num_ideas])
    sources = retrieve_sources(f"{niche} {audience}")
    ideas = await _generate_batch(provider, _prompt(niche, audience, tone, formats, sources), formats)

    initial = list(ideas)
    results: list[dict] = []
    for index, idea in enumerate(ideas):
        errors = validate_idea(idea, niche, audience)
        best, best_errors = idea, errors
        for _ in range(MAX_REFLECTION_ATTEMPTS if errors else 0):
            candidate_prompt = _prompt(
                niche, audience, tone, [formats[index]], sources,
                feedback="; ".join(best_errors),
            )
            try:
                candidate = (await _generate_batch(provider, candidate_prompt, [formats[index]]))[0]
            except ProviderError:
                logger.warning("Targeted regeneration failed for idea %d", index + 1)
                break
            candidate_errors = validate_idea(candidate, niche, audience)
            if _idea_score(candidate_errors) > _idea_score(best_errors):
                best, best_errors = candidate, candidate_errors
            if not candidate_errors:
                break
        results.append(_result_dict(best, best_errors))

    embeddings = await provider.embed([_fingerprint(idea) for idea in initial])
    duplicate_pairs = _duplicate_pairs(initial, embeddings)
    for index, prior_index in duplicate_pairs:
        original_errors = results[index]["validation_errors"]
        best = BlogIdea.model_validate(results[index]["idea"])
        prior = BlogIdea.model_validate(results[prior_index]["idea"])
        best_errors = list(original_errors)
        for _ in range(MAX_REFLECTION_ATTEMPTS):
            prompt = _prompt(
                niche, audience, tone, [formats[index]], sources,
                feedback=(
                    "This idea is too similar to another idea. Choose a clearly different "
                    "specific problem, hook, and angle. " + "; ".join(best_errors)
                ),
            )
            try:
                candidate = (await _generate_batch(provider, prompt, [formats[index]]))[0]
            except ProviderError:
                logger.warning("Duplicate regeneration failed for idea %d", index + 1)
                break
            candidate_errors = validate_idea(candidate, niche, audience)
            candidate_vectors = await provider.embed(
                [_fingerprint(candidate), _fingerprint(prior)]
            )
            similarity = _similarity(
                candidate_vectors[0], candidate_vectors[1]
            ) if candidate_vectors else lexical_similarity(candidate, prior)
            if similarity < DUPLICATE_THRESHOLD and len(candidate_errors) <= len(best_errors):
                best, best_errors = candidate, candidate_errors
                break
        if len(best_errors) <= len(original_errors):
            results[index] = _result_dict(best, best_errors)

    final_duplicates = _duplicate_pairs(
        [BlogIdea.model_validate(item["idea"]) for item in results],
        await provider.embed([_fingerprint(BlogIdea.model_validate(item["idea"])) for item in results]),
    )
    passed = sum(not item["validation_errors"] for item in results)
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    return GenerationResult(
        ideas=results,
        sources=sources,
        mode=provider.mode,
        model=provider.model or _default_model(provider.mode),
        latency_ms=elapsed,
        validation_pass_rate=passed / max(len(results), 1),
        duplicate_rate=len(final_duplicates) / max(len(results), 1),
    )


def _result_dict(idea: BlogIdea, errors: list[str]) -> dict:
    return {
        "idea": idea.model_dump(),
        "validation_errors": errors,
        "valid": not errors,
    }


def _fingerprint(idea: BlogIdea) -> str:
    return " ".join([idea.title, idea.hook, idea.angle, idea.target_keyword])


def _duplicate_pairs(
    ideas: list[BlogIdea], embeddings: list[list[float]] | None
) -> list[tuple[int, int]]:
    duplicates = []
    for right in range(len(ideas)):
        for left in range(right):
            similarity = (
                _similarity(embeddings[left], embeddings[right])
                if embeddings
                else lexical_similarity(ideas[left], ideas[right])
            )
            if similarity >= DUPLICATE_THRESHOLD:
                duplicates.append((right, left))
                break
    return duplicates


def _similarity(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if not denominator or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def lexical_similarity(left: BlogIdea, right: BlogIdea) -> float:
    def vector(idea: BlogIdea) -> Counter[str]:
        return Counter(_terms(_fingerprint(idea)))

    left_vector, right_vector = vector(left), vector(right)
    dot = sum(count * right_vector.get(term, 0) for term, count in left_vector.items())
    left_norm = math.sqrt(sum(count * count for count in left_vector.values()))
    right_norm = math.sqrt(sum(count * count for count in right_vector.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _fit_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > 70:
        title = title[:70].rsplit(" ", 1)[0].rstrip(" ,:;-")
    if len(title) < 50:
        suffix = " | A Practical Guide"
        title = title.rstrip(" .:;-") + suffix
    if len(title) > 70:
        title = title[:70].rsplit(" ", 1)[0].rstrip(" ,:;-")
    return title


def demo_ideas_from_prompt(prompt: str) -> list[dict]:
    match = re.search(
        r'for the niche "(.*?)" and audience "(.*?)".*?Use each assigned format once, in this order: (\[.*?\])',
        prompt,
        re.S,
    )
    if not match:
        return []
    niche, audience, raw_formats = match.groups()
    formats = json.loads(raw_formats)
    tone_match = re.search(r"Tone: (.*)", prompt)
    tone = tone_match.group(1).splitlines()[0] if tone_match else "Use precise, confident language and avoid slang."
    tone_hook = (
        "A clear, no-nonsense starting point"
        if "precise" in tone
        else "A friendly, approachable way to get started"
        if "warmly" in tone
        else "A light-hearted but practical way to get started"
        if "wit" in tone
        else "A motivating, achievable way to get started"
        if "progress" in tone
        else "A clear learning path with explanations you can apply"
    )
    patterns = {
        "listicle": (f"5 {niche} Skills Every {audience} Should Practice Today", "Practice one skill at a time"),
        "how-to": (f"How {audience} Can Build a First {niche} Project", "Turn a small brief into a portfolio-ready project"),
        "comparison": (f"Choosing Your First {niche} Tool: A Guide for {audience}", "Compare tools against practical needs"),
        "case study": (f"A {audience} {niche} Project: From Brief to Portfolio", "Follow a realistic project from question to presentation"),
        "myth-busting": (f"Beyond the Myths: A Practical {niche} Guide for {audience}", "Separate common assumptions from actions you can verify"),
        "interview prep": (f"Preparing for {niche} Interviews: A {audience} Checklist", "Practice explaining decisions, tradeoffs, and results"),
    }
    results = []
    for selected_format in formats:
        title, angle = patterns[selected_format]
        title = _fit_title(title)
        body = (
            [(f"Skill {number}: {topic}", f"Practice {topic.lower()} in a small {niche} task and explain how it helps {audience}.")
             for number, topic in enumerate(("core concepts", "hands-on setup", "data or inputs", "clear communication", "portfolio evidence"), 1)]
            if selected_format == "listicle"
            else [
                ("Comparison criteria", "Define a manageable question and the criteria used to compare the options."),
                ("Option A: strengths and limits", f"Assess one {niche} option against the needs of {audience}."),
                ("Option B: strengths and limits", f"Assess a second {niche} option using the same criteria."),
                ("Choose for your situation", "Explain which tradeoffs matter most and when either option is appropriate."),
            ]
            if selected_format == "comparison"
            else [
                ("The starting brief", f"Introduce a realistic {niche} challenge for {audience}, without inventing outcomes."),
                ("Decisions along the way", "Describe the choices, constraints, and tradeoffs in the example."),
                ("What the process demonstrates", "Separate observable process lessons from claims that would need evidence."),
                ("Adapt the approach", "Show how a reader can apply the method to a smaller practice project."),
            ]
            if selected_format == "case study"
            else [
                ("The common assumption", f"State one specific assumption about {niche} that {audience} may encounter."),
                ("What can be checked", "Identify evidence a reader can look for instead of accepting a broad claim."),
                ("A more useful explanation", "Offer a balanced, practical alternative that acknowledges limits."),
                ("Try a small test", "Suggest a safe way to test the idea in a personal learning exercise."),
            ]
            if selected_format == "myth-busting"
            else [
                ("Read the role requirements", f"Find the {niche} skills and audience context mentioned in an interview brief."),
                ("Practice a technical explanation", "Explain a relevant concept with a small example and clear assumptions."),
                ("Discuss tradeoffs", "Prepare to compare approaches and explain what you would validate next."),
                ("Reflect and improve", f"Turn practice feedback into one concrete next step for {audience}."),
            ]
            if selected_format == "interview prep"
            else [
                ("Define the goal", f"Choose a realistic {niche} problem relevant to {audience} and write down the desired outcome."),
                ("Select the approach", "Compare two reasonable options and explain the tradeoffs before starting."),
                ("Build and test", "Complete a small, repeatable task and check the result against the original goal."),
                ("Explain the work", "Summarize the decisions, limitations, and next improvement without inventing results."),
            ]
        )
        outline = [
            {"heading": "Introduction", "summary": f"Set up the {niche} question and why it matters to {audience}."},
            *[{"heading": heading, "summary": summary} for heading, summary in body],
            {"heading": "Conclusion and next steps", "summary": f"Recap a practical next action for {audience} exploring {niche}."},
        ]
        results.append({
            "title": title,
            "hook": f"{tone_hook} for {audience} exploring {niche}, grounded in practical examples rather than hype.",
            "angle": angle,
            "target_keyword": f"{niche} for {audience}",
            "search_intent": "informational",
            "audience": audience,
            "est_word_count": 1200,
            "format": selected_format,
            "outline": outline,
        })
    return results


async def regenerate_one(
    niche: str, audience: str, tone: str, selected_format: str, feedback: str
) -> GenerationResult:
    started = time.perf_counter()
    provider = get_provider()
    sources = retrieve_sources(f"{niche} {audience}")
    prompt = _prompt(niche, audience, tone, [selected_format], sources, feedback)
    idea = (await _generate_batch(provider, prompt, [selected_format]))[0]
    errors = validate_idea(idea, niche, audience)
    for _ in range(MAX_REFLECTION_ATTEMPTS if errors else 0):
        candidate_prompt = _prompt(
            niche, audience, tone, [selected_format], sources,
            feedback="; ".join(errors),
        )
        try:
            candidate = (await _generate_batch(provider, candidate_prompt, [selected_format]))[0]
        except ProviderError:
            break
        candidate_errors = validate_idea(candidate, niche, audience)
        if len(candidate_errors) < len(errors):
            idea, errors = candidate, candidate_errors
        if not errors:
            break
    return GenerationResult(
        ideas=[_result_dict(idea, errors)],
        sources=sources,
        mode=provider.mode,
        model=provider.model or _default_model(provider.mode),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        validation_pass_rate=float(not errors),
        duplicate_rate=0.0,
    )
