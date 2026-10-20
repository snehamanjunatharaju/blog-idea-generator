# Evaluation

Generated on the selected `demo` provider across 15 sample niches (45 ideas, 3 per niche).

| Niche | Validation pass rate | Near-duplicate rate | Latency (ms) | Cost |
|---|---:|---:|---:|---:|
| Power BI | 100% | 0% | 0.64 | $0.00 |
| Python for data analysis | 100% | 0% | 0.32 | $0.00 |
| personal finance | 100% | 0% | 0.29 | $0.00 |
| home fitness | 100% | 0% | 0.27 | $0.00 |
| sustainable fashion | 100% | 0% | 0.28 | $0.00 |
| beginner photography | 100% | 0% | 0.27 | $0.00 |
| small business marketing | 100% | 0% | 0.27 | $0.00 |
| cloud computing | 100% | 0% | 0.27 | $0.00 |
| healthy meal planning | 100% | 0% | 0.27 | $0.00 |
| career transitions | 100% | 0% | 0.26 | $0.00 |
| interior design | 100% | 0% | 0.26 | $0.00 |
| cybersecurity basics | 100% | 0% | 0.26 | $0.00 |
| product management | 100% | 0% | 0.27 | $0.00 |
| urban gardening | 100% | 0% | 0.27 | $0.00 |
| technical interview preparation | 100% | 0% | 0.27 | $0.00 |
| **Average** | **100%** | **0%** | **0.30** | **$0.00 (Demo mode)** |

## Method and limitations

- The script runs in Demo mode by default and exercises the validators, duplicate checks, and retry pipeline without API keys or network calls. Pass `--provider ollama`, `--provider openai`, or `--provider anthropic` to opt into live model calls.
- Demo pass rate and zero cost measure the deterministic fixture path, not the quality or cost of an LLM. Treat it as a smoke/evaluation harness, not a model-quality benchmark.
- Live provider token usage and prices are not currently captured; live cost is therefore reported as **Not measured**, never estimated.
- Latency is wall-clock time for this run and can vary by machine, model, and network.

Re-run with `python scripts/evaluate.py` to refresh this table. This report is generated output; do not present Demo-mode results as evidence of production model performance.
