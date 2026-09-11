"""Fail the build on any figure the journal cannot print as supplied.

Everything here is measurable. Taste is not checked and is not the point: the
figures were up to 2.6x too wide, which no amount of visual review caught
because each one looked correct on screen at its own scale.

REPRODUCIBILITY IS CHECKED ON THE PNG, NEVER THE PDF. Two identical runs of a
figure script produce PDFs with different bytes at identical size, because
matplotlib's font subsetting is not deterministic; the PNG of the same two runs
is byte-identical. Anyone verifying that a figure rebuilds unchanged must
compare PNGs. Comparing PDFs will report a difference on every rebuild and
train the reader to ignore it, which is how a real change gets waved through.

WHAT THIS DOES NOT CHECK, and must not claim to: rendered font size. Extracting
point sizes from a PDF is unreliable, and a guard that reports a check it does
not perform is worse than no guard. Minimum size is guaranteed by construction
instead. `nag.figstyle.apply_style` sets the smallest rcParam to 6 pt, and
because figures are authored at final printed width nothing is scaled
afterwards, so 6 pt is what prints. A hard-coded `fontsize=` below 5 in a script
is caught only by reading the figure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGS = REPO_ROOT / "output" / "figures"
DOUBLE_MM, SINGLE_MM, TOL_MM = 183.0, 89.0, 1.0


def page_size_mm(pdf: Path) -> tuple[float, float]:
    raw = pdf.read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", raw)
    if not m:
        raise ValueError(f"{pdf.name}: no MediaBox")
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return ((x1 - x0) / 72 * 25.4, (y1 - y0) / 72 * 25.4)


def has_live_text(pdf: Path) -> bool:
    """Outlined text cannot be re-typeset or corrected by the journal."""
    return b"/Font" in pdf.read_bytes()


def main() -> int:
    # `._*` are AppleDouble sidecars: real files on this exFAT volume, and not PDFs.
    pdfs = [p for p in sorted(FIGS.glob("*.pdf")) if not p.name.startswith("._")]
    if not pdfs:
        print(f"no figures found in {FIGS}", file=sys.stderr)
        return 1

    problems = []
    for pdf in pdfs:
        w, h = page_size_mm(pdf)
        ok = abs(w - DOUBLE_MM) < TOL_MM or abs(w - SINGLE_MM) < TOL_MM
        flag = "ok " if ok else "OVER"
        if not ok:
            problems.append(f"{pdf.name}: {w:.0f} mm wide; must be {SINGLE_MM:.0f} mm "
                            f"(single column) or {DOUBLE_MM:.0f} mm (double) at final size")
        if not has_live_text(pdf):
            problems.append(f"{pdf.name}: text is outlined rather than live vector")
        print(f"  {flag}  {w:6.1f} x {h:5.1f} mm  {pdf.name}")

    if problems:
        print("\n" + "\n".join(problems), file=sys.stderr)
        print(f"\n{len(problems)} problem(s). A figure wider than the column is "
              f"scaled down by the journal, shrinking every label with it.",
              file=sys.stderr)
        return 1
    print(f"\n{len(pdfs)} figure(s): every one is a journal column width with live text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
