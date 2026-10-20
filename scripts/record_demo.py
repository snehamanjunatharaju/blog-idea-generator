"""Record representative local Ollama outputs for offline Demo mode."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from idea_engine import ProviderError, generate_ideas, regenerate_one


SAMPLES = [
    ("Power BI", "Power BI for beginners"),
    ("SQL", "SQL practice for beginners"),
    ("Data analytics", "data analytics portfolio"),
    ("Personal finance", "personal finance basics"),
    ("Fitness", "beginner fitness routine"),
    ("Interior design", "small space interior design"),
    ("Cooking for beginners", "cooking basics for students"),
    ("Career development", "career planning for freshers"),
]


async def main() -> None:
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "llama3.1:8b"
    output = Path(__file__).resolve().parents[1] / "fixtures" / "llama31_samples.json"
    recordings = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    for index, (niche, keyword) in enumerate(SAMPLES, start=1):
        existing = next((item for item in recordings if item.get("niche") == niche), None)
        if existing and existing.get("ideas") and existing["ideas"][0].get("valid"):
            print(f"Already valid, keeping: {niche}", flush=True)
            continue
        print(f"Recording {index}/{len(SAMPLES)}: {niche}", flush=True)
        if existing:
            prior = existing["ideas"][0]
            errors = prior.get("validation_errors", [])
            focus = []
            if any("audience" in error.lower() for error in errors):
                focus.append(
                    'Repeat the exact audience phrase "students and freshers in India" in '
                    "the title, hook, angle, and substantive section summaries."
                )
            if any("niche" in error.lower() for error in errors):
                focus.append(f'Use the exact niche phrase "{niche}" throughout the brief.')
            if any("promises" in error.lower() for error in errors):
                focus.append("Do not use a number in the title.")
            if any("introduction" in error.lower() or "conclusion" in error.lower() for error in errors):
                focus.append("Use an Introduction first and a Conclusion last.")
            try:
                result = await regenerate_one(
                    niche,
                    "students and freshers in India",
                    "professional",
                    prior["idea"]["format"],
                    "Fix these validation issues: "
                    + "; ".join(errors)
                    + ". "
                    + " ".join(focus)
                    + " Ensure a 50-70 character title and 3-6 substantive sections.",
                    provider_name="ollama",
                )
            except ProviderError as exc:
                print(f"Could not regenerate {niche}: {type(exc).__name__}", flush=True)
                continue
        else:
            try:
                result = await generate_ideas(
                    niche=niche,
                    audience="students and freshers in India",
                    num_ideas=1,
                    tone="professional",
                    model="llama3.1:8b",
                    provider_name="ollama",
                    target_keyword=keyword,
                )
            except ProviderError as exc:
                print(f"Could not generate {niche}: {type(exc).__name__}", flush=True)
                continue
        new_idea = result.ideas[0]
        if existing and len(existing["ideas"][0].get("validation_errors", [])) < len(new_idea["validation_errors"]):
            new_idea = existing["ideas"][0]
        recording = {
            "source": "Recorded from llama3.1:8b via local Ollama",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "niche": niche,
            "audience": "students and freshers in India",
            "mode": result.mode,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "sources": result.sources,
            "ideas": [new_idea],
        }
        if existing:
            recordings[recordings.index(existing)] = recording
        else:
            recordings.append(recording)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(recordings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Saved {niche}: valid={new_idea['valid']} "
            f"latency_ms={result.latency_ms:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
