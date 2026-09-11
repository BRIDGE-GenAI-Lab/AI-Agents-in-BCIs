"""Renumber figure cross-references for the NBE sequence, in one atomic pass.

The sequence changes to architecture, headline, mechanism, challenge,
consequence:

    Figure 3 caution      -> eFigure 3     (demoted)
    Figure 4 calibration  -> eFigure 4     (demoted)
    Figure 5 naturalistic -> Figure 4
    Figure 6 replay       -> Figure 5
    (new) abstention      -> Figure 3

WHY THIS IS A SCRIPT AND NOT A SEQUENCE OF EDITS. Renaming 5 to 4 while a
different figure is already called 4, and 6 to 5 while another is already 5,
collides. Applied one at a time in the wrong order, "Figure 5" becomes
"Figure 4" and is then caught by the next rule and becomes "Figure 3", silently
merging two figures into one reference. Both files still read as valid English
afterwards, which is why a human proofread does not catch it.

Every replacement therefore goes through a placeholder token first, so no
intermediate state is ever a real figure number.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# figure_legends.md is NOT in this list, and that is deliberate. It was
# renumbered in cc31d6a as part of building the demoted figures, so it already
# reads eFigure 3, eFigure 4, Figure 4, Figure 5. Running the map over it again
# would take the ALREADY-CORRECT "Figure 4" (naturalistic) to "eFigure 4" and
# the already-correct "Figure 5" (replay) to "Figure 4", merging two legends
# onto one number. The result still reads as valid English, which is exactly
# the failure this script exists to prevent, so re-running it here is not safe.
# supplement.md is included: it currently holds only eFigure references, which
# the (?<!e) guard leaves alone, so the pass over it is a no-op that will keep
# working if a main-figure reference is ever added to it.
FILES = [REPO_ROOT / "manuscript" / "manuscript.md",
         REPO_ROOT / "manuscript" / "supplement.md"]

# old label -> new label. Order is irrelevant because of the placeholder pass.
MAP = {
    "Figure 3": "eFigure 3",
    "Figure 4": "eFigure 4",
    "Figure 5": "Figure 4",
    "Figure 6": "Figure 5",
}


def renumber(text: str) -> tuple[str, dict]:
    counts = {}
    # Pass 1: every old label to a unique placeholder that cannot match any rule.
    for i, old in enumerate(MAP):
        # (?<!e) so "eFigure 3" is never matched by the rule for "Figure 3".
        pat = re.compile(rf"(?<!e)\b{re.escape(old)}\b")
        text, n = pat.subn(f"\x00FIG{i}\x00", text)
        counts[old] = n
    # Pass 2: placeholders to their new labels.
    for i, new in enumerate(MAP.values()):
        text = text.replace(f"\x00FIG{i}\x00", new)
    return text, counts


def main() -> int:
    if any("\x00" in f.read_text() for f in FILES):
        print("a source file already contains the placeholder byte; refusing",
              file=sys.stderr)
        return 1
    total = {}
    for f in FILES:
        out, counts = renumber(f.read_text())
        f.write_text(out)
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
        print(f"  {f.name}: " + ", ".join(f"{k}x{v}" for k, v in counts.items() if v))
    print("\ntotals: " + ", ".join(f"{k} -> {MAP[k]}: {v}" for k, v in total.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
