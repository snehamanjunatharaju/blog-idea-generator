import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idea_engine import generate_ideas


SAMPLE_NICHES = [
    "Power BI",
    "Python for data analysis",
    "personal finance",
    "home fitness",
    "sustainable fashion",
    "beginner photography",
    "small business marketing",
    "cloud computing",
    "healthy meal planning",
    "career transitions",
    "interior design",
    "cybersecurity basics",
    "product management",
    "urban gardening",
    "technical interview preparation",
]


async def evaluate(provider_name: str) -> list[dict]:
    os.environ["LLM_PROVIDER"] = provider_name
    results = []
    for niche in SAMPLE_NICHES:
        result = await generate_ideas(
            niche=niche,
            audience="beginners in India",
            num_ideas=3,
            tone="professional",
        )
        results.append({
            "niche": niche,
            "pass_rate": result.validation_pass_rate,
            "duplicate_rate": result.duplicate_rate,
            "latency_ms": result.latency_ms,
            "mode": result.mode,
        })
    return results


def render_report(results: list[dict], provider_name: str) -> str:
    total_ideas = len(results) * 3
    passed_ideas = round(sum(row["pass_rate"] * 3 for row in results))
    avg_pass = passed_ideas / total_ideas if total_ideas else 0
    avg_duplicate = sum(row["duplicate_rate"] for row in results) / len(results)
    avg_latency = sum(row["latency_ms"] for row in results) / len(results)
    cost = "$0.00 (Demo mode)" if all(row["mode"] == "demo" for row in results) else "Not measured"
    lines = [
        "# Evaluation",
        "",
        f"Generated on the selected `{provider_name}` provider across {len(results)} sample niches "
        f"({total_ideas} ideas, 3 per niche).",
        "",
        "| Niche | Validation pass rate | Near-duplicate rate | Latency (ms) | Cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        row_cost = "$0.00" if row["mode"] == "demo" else "Not measured"
        lines.append(
            f"| {row['niche']} | {row['pass_rate']:.0%} | {row['duplicate_rate']:.0%} "
            f"| {row['latency_ms']:.2f} | {row_cost} |"
        )
    lines.extend([
        f"| **Average** | **{avg_pass:.0%}** | **{avg_duplicate:.0%}** | **{avg_latency:.2f}** | **{cost}** |",
        "",
        "## Method and limitations",
        "",
        "- The script runs in Demo mode by default and exercises the validators, duplicate checks, and retry pipeline without API keys or network calls. Pass `--provider ollama`, `--provider openai`, or `--provider anthropic` to opt into live model calls.",
        "- Demo pass rate and zero cost measure the deterministic fixture path, not the quality or cost of an LLM. Treat it as a smoke/evaluation harness, not a model-quality benchmark.",
        "- Live provider token usage and prices are not currently captured; live cost is therefore reported as **Not measured**, never estimated.",
        "- Latency is wall-clock time for this run and can vary by machine, model, and network.",
        "",
        "Re-run with `python scripts/evaluate.py` to refresh this table. This report is generated output; do not present Demo-mode results as evidence of production model performance.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the idea validation pipeline on 15 niches.")
    parser.add_argument(
        "--provider",
        choices=["demo", "ollama", "openai", "anthropic"],
        default="demo",
        help="Provider to use. Defaults to offline Demo mode.",
    )
    parser.add_argument("--output", default="docs/evaluation.md")
    args = parser.parse_args()
    results = asyncio.run(evaluate(args.provider))
    report = render_report(results, args.provider)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved evaluation to {output}")


if __name__ == "__main__":
    main()
