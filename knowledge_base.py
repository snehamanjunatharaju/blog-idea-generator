import re


KNOWLEDGE_BASE = (
    {
        "id": "seo-intent",
        "title": "SEO basics: match the search intent",
        "content": (
            "Choose a specific target keyword and make the format serve the searcher's "
            "intent. Informational content should answer a clear question; avoid promising "
            "rankings or traffic."
        ),
    },
    {
        "id": "headline-specificity",
        "title": "Headline formula: audience + outcome + specificity",
        "content": (
            "A useful headline names its audience and practical outcome. Lists and how-to "
            "headlines must deliver exactly what the title promises."
        ),
    },
    {
        "id": "beginner-pain-points",
        "title": "Beginner audience: reduce first-step uncertainty",
        "content": (
            "Beginners often need a clear starting point, realistic examples, plain-language "
            "definitions, and an honest account of limitations."
        ),
    },
    {
        "id": "outline-variety",
        "title": "Editorial planning: make each outline earn its format",
        "content": (
            "A comparison weighs options against criteria; a case study follows a specific "
            "process; interview prep rehearses explanation and tradeoffs. Avoid reusing a "
            "generic introduction-tips-conclusion template."
        ),
    },
    {
        "id": "grounded-claims",
        "title": "Editorial standards: separate evidence from opinion",
        "content": (
            "Do not invent statistics, market trends, research findings, guarantees, or "
            "industry-wide claims. Label opinions as perspectives and cite evidence when it "
            "is actually supplied."
        ),
    },
)


def retrieve_sources(query: str, limit: int = 3) -> list[dict]:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = sorted(
        KNOWLEDGE_BASE,
        key=lambda item: len(
            tokens & set(re.findall(r"[a-z0-9]+", f"{item['title']} {item['content']}".lower()))
        ),
        reverse=True,
    )
    return [dict(source) for source in ranked[:limit]]
