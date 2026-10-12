"""Published files must not leak local machine paths or account names."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"/home/[a-z_][a-z0-9_-]*/"),
    re.compile(r"/Users/[A-Za-z]"),
]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "runs"}
SKIP_SUFFIXES = {".gz", ".pt", ".png", ".pdf"}


def test_no_local_paths():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix in SKIP_SUFFIXES or path.name == "test_no_local_paths.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += [f"{path.relative_to(ROOT)}: {p.pattern}" for p in PATTERNS if p.search(text)]
    assert not hits, hits
