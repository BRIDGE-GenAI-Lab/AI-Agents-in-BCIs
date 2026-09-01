"""Word counts for the three limits *Nature Machine Intelligence* enforces.

These numbers ship: they appear on the title page and twice in the submission
checklist, and an editor checks them against the file. They were previously
computed by hand and went stale without anything noticing. The main text was
recorded as 3,122 while the file had moved on to 3,179; the recorded Methods
count was within one word, which is what showed the counting rule itself was
right and only the main-text figure had aged.

One rule, applied to all three, matching the checklist's own wording:
Introduction, Results and Discussion, excluding headings, references and
display-item legends. Methods is counted separately because NMI excludes it
from the main-text limit. Pipe-table rows, block quotes, citation markers and
emphasis marks are not words.

    python3 code/scripts/wordcount_nmi.py [manuscript.md]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "manuscript" / "manuscript.md"

LIMITS = {"abstract": 150, "main text": 3500}


def _words(segment: str) -> int:
    total = 0
    for line in segment.split("\n"):
        s = line.strip()
        if not s or s == "---":
            continue
        if s.startswith(("#", "|", ">")):          # headings, tables, quotes
            continue
        if re.match(r"^\*{0,2}(figure|table|efigure|etable)\s", s, re.I):
            continue                                # display-item legend
        s = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", s)   # citation markers
        s = re.sub(r"[*_`]", "", s)                    # emphasis
        total += len(s.split())
    return total


def counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    para: list[str] = []
    for line in lines[lines.index("## Abstract") + 1:]:
        if line.strip() == "---":
            break
        if line.strip():
            para.append(line.strip())

    body = text.index("---", text.index("## Abstract"))
    methods = text.index("## Methods")
    refs = text.index("## References")
    return {
        "abstract": len(" ".join(para).split()),
        "main text": _words(text[body:methods]),
        "Methods": _words(text[methods:refs]),
    }


# Files that state the counts in prose. Every one of them has gone stale at
# least once, because a number written by hand does not move when the text does.
CLAIM_FILES = [
    REPO_ROOT / "manuscript" / "title_page.md",
    REPO_ROOT / "_submission_ready" / "study_bci_agent_oversight" / "2_Title_Page.md",
    REPO_ROOT / "_submission_ready" / "study_bci_agent_oversight" / "6_Submission_Checklist.md",
]


def check_claims(actual: dict) -> list[str]:
    """Every count asserted in packet prose must match a fresh count.

    This exists because the main-text figure has now gone stale three times:
    recorded as 3,122 while the file said 3,179, then 3,179 while it said
    3,191. Each time it shipped in four places and each time it was found by
    accident. `check_packet_docx.py` cannot see this: the .docx faithfully
    renders whatever wrong number the .md holds, so freshness and coverage
    both pass. A claim about a file is not checked by comparing that file to
    itself.
    """
    import re

    problems = []
    # Any "<n> words" or "Abstract <n>" style figure in these files must be one
    # of the current counts, or a journal LIMIT (which is not a claim about us).
    limits = {150, 3500}
    current = set(actual.values()) | limits
    for f in CLAIM_FILES:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"(?:Abstract|Main text|Methods|main text|abstract)[: ]+"
                             r"(\d{1,3}(?:,\d{3})*)", text):
            n = int(m.group(1).replace(",", ""))
            if n not in current:
                line = text[:m.start()].count("\n") + 1
                problems.append(
                    f"{f.relative_to(REPO_ROOT)}:{line} states {m.group(1)}, which is "
                    f"neither a current count {sorted(actual.values())} nor a limit "
                    f"{sorted(limits)}. Refresh it.")
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    over = []
    for name, n in counts(path).items():
        cap = LIMITS.get(name)
        if cap is None:
            print(f"  {n:>6,}  {name} (counted separately; no limit)")
        else:
            flag = "OVER" if n > cap else "ok"
            print(f"  {n:>6,}  {name} (limit {cap:,}) {flag}")
            if n > cap:
                over.append(f"{name}: {n:,} > {cap:,}")
    stale = check_claims(counts(path)) if path == DEFAULT else []
    for problem in stale:
        print(f"  STALE CLAIM  {problem}")
    if over:
        print("\nOVER THE LIMIT: " + "; ".join(over), file=sys.stderr)
        return 1
    if stale:
        print(f"\n{len(stale)} stale word-count claim(s) in packet prose.", file=sys.stderr)
        return 1
    print("\nPacket prose agrees with a fresh count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
