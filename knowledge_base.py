import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

NICHE_FILES = {
    "power-bi": ("power bi", "powerbi", "business intelligence"),
    "sql": ("sql", "database", "databases"),
    "data-analytics": ("data analytics", "data analysis", "analytics"),
    "personal-finance": ("personal finance", "finance", "budgeting", "saving"),
    "fitness": ("fitness", "workout", "exercise", "wellness"),
    "interior-design": ("interior design", "interiors", "home decor", "decor"),
}


def _read_note(path: Path, *, fallback: bool = False) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    title, _, content = text.partition("\n")
    title = title.lstrip("# ").strip()
    return {
        "id": path.stem,
        "title": f"Generic fallback: {title}" if fallback else title,
        "content": content.strip(),
    }


def _load_niche_notes() -> dict[str, dict]:
    return {
        niche: _read_note(KNOWLEDGE_DIR / f"{niche}.md")
        for niche in NICHE_FILES
        if (KNOWLEDGE_DIR / f"{niche}.md").is_file()
    }


NICHE_NOTES = _load_niche_notes()
GENERIC_NOTES = [
    _read_note(KNOWLEDGE_DIR / "editorial-basics.md", fallback=True)
]


def retrieve_sources(query: str, limit: int = 3) -> list[dict]:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    selected = []
    for slug, aliases in NICHE_FILES.items():
        if any(alias in query.lower() for alias in aliases) and slug in NICHE_NOTES:
            selected.append(NICHE_NOTES[slug])
    if not selected:
        selected = GENERIC_NOTES
    ranked = sorted(
        selected,
        key=lambda item: len(
            tokens & set(re.findall(r"[a-z0-9]+", f"{item['title']} {item['content']}".lower()))
        ),
        reverse=True,
    )
    return [dict(source) for source in ranked[:limit]]
