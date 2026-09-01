"""Fail if any packet .docx has drifted from the .md it was built from.

The journal reads the .docx. The .md is what everyone edits and verifies. On
2026-09-01 that gap shipped a KNOWN error twice within one hour: a manuscript
fix landed in 3_Manuscript.md and never reached 3_Manuscript.docx, and an
archival note landed in the plan's .md and never reached its .docx. Both were
caught by a human comparing mtimes, which is not a control.

Nothing else in this project detects this. Every other guard checks the source
of truth: the number audit reads the .md, the de-AI scanner reads the .md, and
`git diff` says the tree is clean because both files are committed. A stale
.docx passes all of them while being the only file that actually ships.

Three checks, because no two of them are sufficient:

  FRESHNESS. A .docx older than its .md is stale by construction. Cheap, and it
  is what caught both real instances.

  CONTENT. A superseded string must not survive in any .docx. Freshness alone
  would pass a .docx rebuilt from a stale .md, and a rebuild can silently drop
  content. SUPERSEDED lists strings that were corrected late, which is exactly
  where a stale artifact hides one.

  COVERAGE. Every substantial paragraph of the .md must actually appear in the
  .docx. This exists because the freshness check has a hole that was found by
  walking into it: `git checkout -- some.docx` sets the file's mtime to NOW, so
  a REVERTED .docx is newer than its .md while its content is a version older.
  Freshness passes, the superseded-string list passes because the reverted text
  predates those corrections, and the shipping file is silently behind. The same
  hole swallows any restore, copy, or touch. Comparing text to text does not
  care how the bytes got there, and it additionally catches a rebuild that
  dropped a section, which neither other check can see.

EXTRACTION IS NOT NAIVE TAG-STRIPPING, and that matters. Removing every tag
concatenates adjacent table cells, so a row ending 14279 | 176 | 7.70 becomes
"...1427917677.7...", which contains "77.7" and produces a false positive on a
superseded string that is not there. This joins <w:t> runs per paragraph and
separates paragraphs, which is what a reader sees.

    python3 code/scripts/check_packet_docx.py     # exit 1 on any problem
"""
from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "_submission_ready" / "study_bci_agent_oversight"

# Strings corrected late enough that a stale artifact could still carry them.
# Each entry is (needle, what it was replaced by), so a failure explains itself
# instead of just naming a string.
SUPERSEDED = [
    ("2 of the 100 sampled", "19 of the 1,084 usable episodes (the pilot count was "
                             "attributed to the primary analysis)"),
    ("-0.505", "caution wording 1's CI is -0.425 to -0.287"),
    ("eleven of twelve", "ten of 12 wordings; TWO reach significance"),
    ("all adjusted *P* > .97", "the smallest adjusted P among the non-significant is .13"),
    ("44.9", "the bare-rate reduction is up to 22 percentage points on the full pool"),
    ("16,200 rows", "50,230 episode runs across six datasets"),
    ("eFigure 4", "renumbered to eFigure 2 (the archived plan is exempt)"),
]
# The archived plan legitimately records what was planned, including eFigure 4.
CONTENT_EXEMPT = {"NMI_Review_Response_Plan.docx"}


def docx_text(path: Path, drop_superscripts: bool = False) -> str:
    """Visible text, joined per paragraph so adjacent cells cannot merge.

    `drop_superscripts` removes runs marked vertAlign=superscript. Those are the
    rendered citation markers: `[1,2]` in the source becomes a superscript "1,2"
    sitting flush against the preceding word, so a paragraph compared against a
    source with its markers stripped never matches. Dropping exactly those runs
    is precise, and leaves every other digit in place, which matters because the
    numbers are the point. The superseded-string scan keeps them.
    """
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (name.startswith("word/") and name.endswith(".xml")):
                continue
            raw = z.read(name).decode("utf8", "ignore")
            for para in re.findall(r"<w:p[ >].*?</w:p>", raw, re.S):
                runs = re.findall(r"<w:r[ >].*?</w:r>", para, re.S)
                if not runs:
                    runs = [para]
                buf = []
                for run in runs:
                    if drop_superscripts and 'w:val="superscript"' in run:
                        continue
                    buf.append("".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", run, re.S)))
                out.append("".join(buf))
    return "\n".join(out)


# Paragraphs shorter than this are skipped: headings, table rows, one-line
# labels and list fragments reflow or get restructured by pandoc, and chasing
# them produces false alarms rather than caught staleness.
MIN_PARAGRAPH_WORDS = 25


def _normalise(text: str) -> str:
    """Collapse the differences pandoc legitimately introduces.

    Curly quotes, en/em dashes, non-breaking spaces and whitespace runs all
    differ between the .md source and the rendered .docx without any content
    having changed. Everything else is compared literally.

    Bracketed citation markers are dropped from BOTH sides. Pandoc leaves `[1,2]`
    as literal text rather than rendering it, so a marker stripped from only the
    source can never match; dropping it symmetrically compares the prose and
    leaves every other digit intact.

    Two more differences are symmetric for the same reason. Word XML escapes
    apostrophes, quotes and angle brackets as entities, so the extracted text
    must be unescaped or every contraction mismatches. And `_` cannot be
    stripped from the source alone: a tool name like `place_call` would become
    `placecall` on one side and stay `place_call` on the other. Emphasis markers
    go from both sides, which costs nothing because emphasis is not content.
    """
    text = html.unescape(text)
    text = re.sub(r"[*_`]", "", text)
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2013", "-").replace("\u2014", "-")
                .replace("\u00a0", " "))
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def md_paragraphs(md: Path) -> list[str]:
    """Substantial prose paragraphs of a markdown file, as plain text."""
    body = md.read_text()
    body = re.sub(r"```.*?```", "", body, flags=re.S)      # fenced code
    out = []
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block or block.startswith(("#", "|", ">", "---", "!")):
            continue
        # A bullet list is one markdown block but N docx paragraphs, so an
        # unsplit list can never match and reports as missing content.
        for item in re.split(r"\n(?=\s*(?:[-*+]|\d+\.)\s)", block):
            plain = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", item.strip())
            plain = re.sub(r"\[[\d,\s]+\]", "", plain)       # citation markers
            plain = re.sub(r"\^[^\s^]*\^", "", plain)          # ^superscript^
            plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", plain)  # links
            plain = re.sub(r"[\\\\]", "", plain)                    # escapes
            plain = _normalise(plain)
            if len(plain.split()) >= MIN_PARAGRAPH_WORDS:
                out.append(plain)
    return out


def check_figures() -> list[str]:
    """Every figure the packet cites must exist, and every file must be cited.

    This exists because the guard below passed cleanly on a packet that was
    missing Figure 6. The manuscript cited it, the legend described it, and the
    checklist counted six figures, but only five files had been copied into
    `figures/`. Nothing noticed: the .md and .docx agreed with each other
    perfectly, which is all the other checks compare. A submission is the whole
    folder, not the prose.

    The reverse direction matters too. An uncited file is either a figure whose
    citation was lost or a stale leftover from a previous numbering, and both
    ship badly.
    """
    figdir = PACKET / "figures"
    if not figdir.is_dir():
        return [f"no figures directory at {figdir}"]

    text = ""
    for name in ("3_Manuscript.md", "5_Figure_Legends.md"):
        f = PACKET / name
        if f.is_file():
            text += f.read_text(encoding="utf-8")

    cited = {m.group(0).replace(" ", "") for m in re.finditer(r"(?<!e)Figure \d+", text)}
    cited |= {m.group(0).replace(" ", "") for m in re.finditer(r"eFigure \d+", text)}
    have = {f.stem for f in figdir.glob("*.pdf") if not f.name.startswith("._")}

    problems = []
    for name in sorted(cited - have):
        problems.append(f"FIGURE MISSING: {name} is cited but {name}.pdf is not in figures/.")
    for name in sorted(have - cited):
        problems.append(f"FIGURE UNCITED: figures/{name}.pdf is in the packet but nothing "
                        f"cites it. Either a citation was lost or the file is stale.")
    if not problems:
        print(f"  ok  figures/ ({len(have)} files, every citation matched)")
    return problems


def main() -> int:
    if not PACKET.is_dir():
        print(f"no packet at {PACKET}", file=sys.stderr)
        return 1

    # `._*` are macOS AppleDouble sidecars. This volume is exFAT, which has no
    # resource forks, so macOS writes them as REAL files that match every glob
    # and are not the type they appear to be: `._3_Manuscript.docx` is a 4 KB
    # blob, not a zip, and opening it raises BadZipFile. This project's ledger
    # records the same trap in the parquet globs; a glob written here without
    # the guard hit it again immediately.
    pairs = [(md, md.with_suffix(".docx"))
             for md in sorted(PACKET.rglob("*.md"))
             if not md.name.startswith("._") and md.with_suffix(".docx").exists()]
    if not pairs:
        print("no .md/.docx pairs found; nothing to check", file=sys.stderr)
        return 1

    problems = []
    for md, dx in pairs:
        rel = dx.relative_to(PACKET)
        if md.stat().st_mtime > dx.stat().st_mtime:
            problems.append(f"STALE: {rel} is older than {md.name}. Rebuild it.")
        if dx.name in CONTENT_EXEMPT:
            print(f"  ok (content exempt)  {rel}")
            continue
        text = docx_text(dx)
        for needle, replaced_by in SUPERSEDED:
            if needle in text:
                problems.append(
                    f"SUPERSEDED STRING in {rel}: {needle!r} still present. "
                    f"It was replaced by: {replaced_by}."
                )
        haystack = _normalise(docx_text(dx, drop_superscripts=True))
        paras = md_paragraphs(md)
        absent = [p for p in paras if p not in haystack]
        if absent:
            problems.append(
                f"CONTENT MISSING from {rel}: {len(absent)} of {len(paras)} "
                f"substantial paragraph(s) in {md.name} do not appear in the "
                f".docx. It was not rebuilt from the current .md. First missing: "
                f"{absent[0][:110]!r}"
            )
        print(f"  ok  {rel}  ({len(paras)} paragraphs matched)")

    problems.extend(check_figures())

    if problems:
        print("\n" + "\n".join(problems), file=sys.stderr)
        print(f"\n{len(problems)} problem(s). The .docx is the file the journal reads.",
              file=sys.stderr)
        return 1
    print(f"\n{len(pairs)} pair(s) checked: every .docx is at least as new as its "
          f".md, carries no superseded string, and contains every substantial "
          f"paragraph of its source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
