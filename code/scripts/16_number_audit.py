"""Number audit: extract every numeric literal from the manuscript and the
supplement, and check that each one is reconcilable with a value that actually
exists in the study's own results tables.

This script exists because two numerical errors reached a submitted draft
without being caught by anyone on the team, only by an external reviewer. The
Results text once said a caution wording was excluded "in all 5 models" when
the supplement's own exclusion table listed 4; an AURC paragraph said "the two
arms below the gate" and then named three. This tool cannot catch either of
those two errors -- see the LIMITATIONS note at the bottom of this docstring --
but it does catch the more common failure mode: a rate, a count, or a
confidence interval bound that was mistyped, stale, or pulled from the wrong
row when the prose was drafted.

Method
------
1. Extract every numeric literal from `manuscript/manuscript.md` and
   `manuscript/supplement.md`, skipping a short, explicit list of things that
   are numbers in the text but are not data (see IGNORE RULES below).
2. Build a "haystack" of every numeric value that exists anywhere in
   `output/tables/*.csv`, `output/tables/*.json`, `output/results_digest.json`,
   and `output/stats_digest.json` -- the sources the task names -- plus two
   frozen instrument files the supplement quotes numbers from directly
   (`code/nag/frozen_prompts.json`, `code/nag/frozen_mapping.json`). Excluding
   those two would make every wording-count and codebook number the
   supplement inlines read as a false UNMATCHED.
3. For each extracted literal, look for a haystack value close enough to
   count as the same number once ordinary prose rounding is allowed for.
   Four progressively looser tolerances are tried in order (exact, rounds to
   the same displayed precision, rounds to double that tolerance, or agrees
   to within 1% relative) and the first one that succeeds is recorded.
   Percent-vs-fraction confusion (53.7% vs 0.537) is handled by trying the
   literal's value, that value divided by 100, and that value multiplied by
   100, and keeping whichever comparison is closest.
4. Write `output/tables/number_audit.csv` with columns `value, context,
   found_in, status`, `status` in `{MATCHED, UNMATCHED, IGNORED}`.

IGNORE RULES (kept short and explicit; every ignored literal is still written
to the CSV as IGNORED with the reason folded into `found_in`, and every
ignored literal outside the References section is also printed to stdout, so
a reader can check the filter is not hiding a real error):

  REFERENCES_SECTION   Anything after the "## References" heading: citation
                        years, volume/issue/page numbers, DOI suffixes.
  CITATION_MARKER      A bracketed inline citation list, e.g. "[10,11]".
  SECTION_LABEL        A number right after Table/Figure/eTable/eFigure/
                        eMethods/Ruling/Task, e.g. "Figure 4A", "eMethods S1".
  TIER_LABEL           1, 2, or 3 right after the word "tier".
  VERSION_SLUG         A number embedded in a hyphenated identifier, e.g. the
                        "5.6" in "gpt-5.6-luna" or the "5" in "claude-sonnet-5".
  PACKAGE_VERSION      A number right after a named software package
                        (Python, NumPy, pandas, SciPy, scikit-learn, PyArrow).
  PVALUE_OR_THRESHOLD  A number directly after a comparison operator (< > <=
                        >=), which is almost always a reporting threshold
                        ("*P* < .001", "adjusted *P* > .99") rather than a
                        tabulated value.
  DEFINED_THRESHOLD    A number whose nearby context names it as a threshold,
                        floor, limit, ceiling, or cap (e.g. "15% ... limit",
                        "coverage floor of 0.10").
  DATE_TOKEN           A bare 4-digit year (1900-2100), or a day-of-month
                        immediately followed by a month name.
  EMBEDDED_IN_IDENTIFIER  A digit run directly touching a letter with no
                        separator, e.g. the "920" inside a truncated SHA
                        display like "920e31ef..." or inside a commit hash.
  CODE_FENCE            A number inside a ``` fenced code block, which is a
                        verbatim dump of a source file (a JSON schema or a
                        frozen prompt/mapping file) and is checked for
                        fidelity by construction, not by numeric lookup.

LIMITATIONS (read before trusting a clean run):
  This script only checks numeric VALUES against tabulated numeric values. It
  cannot check a COUNTING CLAIM stated in words. "in all 5 models" is wrong
  because only 4 of 5 models actually excluded that wording -- but "5" is
  independently correct (there are 5 models in the panel), so it matches
  trivially and the sentence's real error is invisible to this tool. "the two
  arms below the gate" followed by a list of three items has the same blind
  spot from the other direction: "two" is spelled out, not a digit, so it is
  never extracted at all. A clean number-audit run is evidence that the
  DIGITS in the manuscript are traceable to source data; it is not evidence
  that every ENUMERATION or SUBSET CLAIM built out of those digits is
  internally consistent. That still has to be read by a human.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_FILES = [
    REPO_ROOT / "manuscript" / "manuscript.md",
    REPO_ROOT / "manuscript" / "supplement.md",
]
TABLES_DIR = REPO_ROOT / "output" / "tables"
DIGEST_FILES = [
    REPO_ROOT / "output" / "results_digest.json",
    REPO_ROOT / "output" / "stats_digest.json",
]
EXTRA_JSON_FILES = [
    REPO_ROOT / "code" / "nag" / "frozen_prompts.json",
    REPO_ROOT / "code" / "nag" / "frozen_mapping.json",
]
OUT_CSV = TABLES_DIR / "number_audit.csv"

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
PACKAGE_NAMES = ("Python", "NumPy", "pandas", "SciPy", "scikit-learn", "PyArrow")
LABEL_WORDS = ("eTable", "eFigure", "eMethods", "Table", "Figure", "Ruling", "Task")

# Matches a plain integer of any length, a comma-grouped thousands integer
# (16,200), a decimal, or a leading-dot decimal (".001", as p-values are
# conventionally written), with an optional leading sign, leading '$', and
# trailing '%'. The comma-grouped alternative is tried first so "16,200"
# matches in full rather than the bare-digit alternative stopping at "16";
# the bare-digit alternative uses \d+ (not \d{1,3}) so a plain 4+ digit
# integer with no thousands separator (e.g. "3388", "1065") is not silently
# truncated to its first three digits.
NUM_RE = re.compile(
    r"(?P<sign>[-+])?"
    r"(?P<dollar>\$)?"
    r"(?P<body>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>%)?"
)


def _is_alnum(ch: str) -> bool:
    return ch.isalnum()


def find_slug_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of hyphenated identifiers such as model slugs
    ("gpt-5.6-luna", "claude-sonnet-5", "z-ai/glm-5.3-flash") and versioned
    tokens, so a number embedded in one is never treated as free-standing
    data."""
    spans = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9./]*(?:-[A-Za-z0-9./]+)+", text):
        spans.append(m.span())
    return spans


def extract_numbers(text: str) -> list[dict]:
    """Every numeric-literal candidate in `text`, with its exact substring,
    parsed float value, decimal-precision, percent/dollar flags, character
    offsets, and a short surrounding-text context."""
    out = []
    slug_spans = find_slug_spans(text)
    for m in NUM_RE.finditer(text):
        start, end = m.start(), m.end()
        sign, dollar, body, pct = m.group("sign"), m.group("dollar"), m.group("body"), m.group("pct")

        # A '-' directly glued to a preceding letter/digit is a hyphen inside
        # a token (e.g. "gpt-5.6"), not a minus sign -- back the match off it.
        if sign == "-" and start > 0 and _is_alnum(text[start - 1]):
            sign = None
            start += 1

        raw = text[start:end]
        digits = body.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        if sign == "-":
            value = -value
        decimals = len(digits.split(".")[1]) if "." in digits else 0

        in_slug = any(s <= start and end <= e for s, e in slug_spans)

        ctx_start = max(0, start - 45)
        ctx_end = min(len(text), end + 45)
        context = " ".join(text[ctx_start:ctx_end].split())

        # "percentage points" is this manuscript's usual way of writing a
        # percent-scale value without a literal '%' -- including elliptically
        # ("38.1 percentage points were X and 8.2 were Y", where the second
        # number's unit is never repeated), so a number is treated as
        # percent-scale if that phrase appears in either direction within a
        # wider local window, not only immediately after it.
        pct_window = text[max(0, start - 90):min(len(text), end + 90)]
        is_percent = bool(pct) or "percentage point" in pct_window.lower()

        out.append({
            "raw": raw, "value": value, "decimals": decimals,
            "is_percent": is_percent, "has_dollar": bool(dollar),
            "start": start, "end": end, "in_slug": in_slug,
            "context": context, "before": text[max(0, start - 20):start],
            "after": text[end:end + 20],
        })
    return out


def find_code_fence_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"```.*?```", text, re.DOTALL)]


def classify_ignore(entry: dict, full_text: str, references_start: int | None,
                     fence_spans: list[tuple[int, int]]) -> str | None:
    """Return an ignore-reason code, or None if the literal should be matched."""
    start = entry["start"]
    if references_start is not None and start >= references_start:
        return "REFERENCES_SECTION"

    if any(s <= start and entry["end"] <= e for s, e in fence_spans):
        return "CODE_FENCE"

    before, after = entry["before"], entry["after"]

    # EMBEDDED_IN_IDENTIFIER: a digit run directly touching a letter with no
    # separator -- almost always a truncated SHA-256 display or a commit
    # hash (e.g. the "920" inside "920e31ef..."), never free-standing data.
    if (before and before[-1].isalpha()) or (after and after[0].isalpha()):
        return "EMBEDDED_IN_IDENTIFIER"

    # CITATION_MARKER: sits inside a "[...]" span containing only digits,
    # commas, and spaces (e.g. "[10,11]").
    close = full_text.find("]", entry["end"], entry["end"] + 6)
    openb = full_text.rfind("[", max(0, entry["start"] - 8), entry["start"])
    if close != -1 and openb != -1:
        between = full_text[openb + 1:close]
        if re.fullmatch(r"[\d,\s]+", between):
            return "CITATION_MARKER"

    if entry["in_slug"]:
        return "VERSION_SLUG"

    stripped_before = before.rstrip()
    if any(stripped_before.endswith(w) or stripped_before.endswith(w + " S") for w in LABEL_WORDS):
        return "SECTION_LABEL"

    if entry["value"] in (1, 2, 3) and stripped_before.lower().endswith("tier"):
        return "TIER_LABEL"

    if any(stripped_before.endswith(p) for p in PACKAGE_NAMES):
        return "PACKAGE_VERSION"

    op_tail = stripped_before[-2:].strip()
    if op_tail and op_tail[-1] in "<>" :
        return "PVALUE_OR_THRESHOLD"

    window = full_text[max(0, entry["start"] - 40):entry["end"] + 15].lower()
    if any(w in window for w in ("threshold", "floor", "limit", "ceiling", " cap ", "cap.", "cap,")):
        return "DEFINED_THRESHOLD"

    if entry["decimals"] == 0 and not entry["is_percent"] and 1900 <= entry["value"] <= 2100 \
            and entry["value"] == int(entry["value"]):
        # Bare 4-digit year, not glued to more digits (e.g. not part of "20260828").
        if not (before and before[-1].isdigit()) and not (after and after[:1].isdigit()):
            return "DATE_TOKEN"

    after_word = after.strip().split(" ")[0].rstrip(",.")
    if after_word in MONTHS:
        return "DATE_TOKEN"

    return None


# --------------------------------------------------------------------------
# Haystack: every numeric value in the source-of-truth files.

def load_csv_numbers(path: Path) -> list[tuple[float, str]]:
    out = []
    df = pd.read_csv(path)
    label_prefix = path.name
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        for v in series.dropna().tolist():
            out.append((float(v), f"{label_prefix}:{col}"))
    return out


def flatten_json_numbers(obj, path: str, out: list[tuple[float, str]]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and obj != obj:  # NaN
            return
        out.append((float(obj), path))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten_json_numbers(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten_json_numbers(v, f"{path}[{i}]", out)


def build_haystack() -> list[tuple[float, str]]:
    haystack: list[tuple[float, str]] = []
    csv_files = sorted(f for f in TABLES_DIR.glob("*.csv")
                        if not f.name.startswith("._") and f != OUT_CSV)
    json_files = sorted(f for f in TABLES_DIR.glob("*.json") if not f.name.startswith("._"))
    for f in csv_files:
        haystack.extend(load_csv_numbers(f))
    for f in json_files + DIGEST_FILES + EXTRA_JSON_FILES:
        if not f.exists():
            continue
        obj = json.loads(f.read_text())
        flatten_json_numbers(obj, f.name, haystack)
    print(f"Haystack built from {len(csv_files)} CSV files and "
          f"{len(json_files) + len(DIGEST_FILES) + len(EXTRA_JSON_FILES)} JSON files "
          f"under output/tables/, output/, and code/nag/: "
          f"{len(haystack)} numeric values total.")
    for f in csv_files:
        print(f"  csv:  {f.relative_to(REPO_ROOT)}")
    for f in json_files + DIGEST_FILES + EXTRA_JSON_FILES:
        if f.exists():
            print(f"  json: {f.relative_to(REPO_ROOT)}")
    return haystack


# --------------------------------------------------------------------------
# Matching.

TOLERANCE_TIERS = ("exact", "display-rounding", "2x-display-rounding", "1pct-relative")


def tolerance_for(tier: str, decimals: int, target: float) -> float:
    if tier == "exact":
        return 1e-9
    if tier == "display-rounding":
        return 0.5 * 10 ** (-decimals) + 1e-9
    if tier == "2x-display-rounding":
        return 1.0 * 10 ** (-decimals) + 1e-9
    if tier == "1pct-relative":
        return max(1e-3, 0.01 * abs(target))
    raise ValueError(tier)


def best_match(entry: dict, haystack: list[tuple[float, str]]) -> tuple[str | None, str | None]:
    """Return (tolerance_tier, found_in) for the best match, or (None, None).

    The percent<->fraction conversion is tried ONLY when the literal actually
    carried a '%' sign. An earlier version of this function tried v/100 for
    every literal in [0, 100] regardless of whether it was written as a
    percentage, on the theory that it would catch more genuine matches. In
    practice it did the opposite: dividing an already-fraction-scale number
    like a CI bound (-0.631) by 100 produces a tiny, coincidence-prone target
    that lands within tolerance of SOME unrelated cell across a haystack of
    ~15,000 values essentially by chance (verified: -0.631/100 = -0.00631
    landed within 0.0005 of a caution-wording risk difference that has
    nothing to do with the sentence being checked). Every proportion column
    in this study's own tables is already on a 0-1 scale, never 0-100, so
    there is no genuine case here that needs the reverse (x100) conversion
    at all, and the /100 conversion is only ever legitimate when the source
    text itself signalled "this is a percentage" with a '%'.
    """
    v = entry["value"]
    variants = [(v, "as-written")]
    if entry["is_percent"]:
        variants.append((v / 100.0, "as-fraction-of-percent"))

    for tier in TOLERANCE_TIERS:
        best = None  # (abs_diff, source_label, variant_label)
        for variant_value, variant_label in variants:
            tol = tolerance_for(tier, entry["decimals"], variant_value)
            for target, label in haystack:
                diff = abs(variant_value - target)
                if diff <= tol and (best is None or diff < best[0]):
                    best = (diff, label, variant_label)
        if best is not None:
            _, label, variant_label = best
            suffix = "" if variant_label == "as-written" else f", {variant_label}"
            return tier, f"{label} (tol={tier}{suffix})"
    return None, None


# --------------------------------------------------------------------------


# --- enumeration claims: the class this tool structurally cannot verify -----

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def enumeration_claims(text: str) -> list[dict]:
    """Surface sentences that state a COUNT and then list the things counted.

    Both errors that motivated this script are invisible to the numeric audit
    above, and no amount of tolerance tuning would have caught either:

      "one caution wording in all 5 models"  -- 5 matches a real value (there
      are five models), so the literal audit passes. The claim is still false;
      the wording was excluded in four of them.

      "The two arms below the gate were A, B, and C"  -- "two" is a word, never
      extracted as a digit, and the three listed items are prose. Nothing to
      match against.

    A clean numeric audit is therefore NOT proof of correctness. It verifies
    that quoted values exist somewhere in the outputs; it says nothing about
    whether a counting or enumeration claim is true. This function does not
    verify these claims either. It finds them and prints them so a human reads
    every one, which is the only check that actually works on this class.
    """
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        m = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\b\s+\w+", low)
        digit = re.search(r"\b(\d{1,3})\s+(?:of|models?|arms?|cells?|wordings?|"
                          r"participants?|studies|episodes?)\b", low)
        if not (m or digit):
            continue
        # A list follows if the sentence has commas or an "and" joining items.
        if (" and " in low or low.count(",") >= 1) and any(
                w in low for w in ("arm", "model", "cell", "wording", "study",
                                    "participant", "episode", "condition")):
            stated = NUMBER_WORDS.get(m.group(1)) if m else int(digit.group(1))
            out.append({"stated_count": stated, "sentence": sent.strip()[:300]})
    return out


def main() -> None:
    haystack = build_haystack()

    rows = []
    ignored_counts: dict[str, int] = {}
    ignored_examples: list[dict] = []
    matched = 0
    unmatched_rows = []

    for path in MANUSCRIPT_FILES:
        text = path.read_text()
        ref_match = re.search(r"^##\s*References\s*$", text, re.MULTILINE)
        references_start = ref_match.start() if ref_match else None
        fence_spans = find_code_fence_spans(text)

        for entry in extract_numbers(text):
            reason = classify_ignore(entry, text, references_start, fence_spans)
            label = f"{path.name}: {entry['context']}"
            if reason is not None:
                ignored_counts[reason] = ignored_counts.get(reason, 0) + 1
                row = {"value": entry["raw"], "context": label,
                       "found_in": f"IGNORED: {reason}", "status": "IGNORED"}
                rows.append(row)
                if reason != "REFERENCES_SECTION":
                    ignored_examples.append({**row, "reason": reason})
                continue

            tier, found_in = best_match(entry, haystack)
            if tier is not None:
                matched += 1
                rows.append({"value": entry["raw"], "context": label,
                             "found_in": found_in, "status": "MATCHED"})
            else:
                row = {"value": entry["raw"], "context": label,
                       "found_in": "", "status": "UNMATCHED"}
                rows.append(row)
                unmatched_rows.append(row)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["value", "context", "found_in", "status"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ["value", "context", "found_in", "status"]})

    n_ignored = sum(ignored_counts.values())
    print()
    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}: {len(rows)} numeric literals "
          f"({matched} MATCHED, {len(unmatched_rows)} UNMATCHED, {n_ignored} IGNORED).")
    print()
    print("IGNORED, by reason:")
    for reason, count in sorted(ignored_counts.items()):
        print(f"  {reason}: {count}")
    print()
    print("IGNORED literals outside the References section (verify the filter is not hiding a real error):")
    for row in ignored_examples:
        print(f"  [{row['reason']}] {row['value']!r} -- {row['context']}")

    print()
    if unmatched_rows:
        print(f"UNMATCHED ({len(unmatched_rows)}) -- each is a submission blocker until resolved:")
        for row in unmatched_rows:
            print(f"  {row['value']!r} -- {row['context']}")
    else:
        print("UNMATCHED: none.")

    # The class this tool cannot verify. Printed last so it is the final thing
    # a reader sees, because a clean UNMATCHED list invites exactly the false
    # confidence that let two counting errors reach a submitted draft.
    claims = []
    for path in MANUSCRIPT_FILES:
        for c in enumeration_claims(path.read_text()):
            c["document"] = path.name
            claims.append(c)
    print()
    print("=" * 72)
    print(f"ENUMERATION CLAIMS REQUIRING A HUMAN TO READ THEM: {len(claims)}")
    print("The numeric audit above CANNOT verify these, and both errors that")
    print("motivated this script were of exactly this kind:")
    print("  'one caution wording in all 5 models' -- 5 is a real value, so it MATCHED,")
    print("     and the claim was still false.")
    print("  'The two arms below the gate were A, B, and C' -- 'two' is a word, never")
    print("     extracted, and the three items are prose.")
    print("A clean audit is not proof of correctness. Read every line below.")
    print("=" * 72)
    for c in claims:
        print(f"  [{c['document']}] states {c['stated_count']}: {c['sentence']}")
    if claims:
        pd.DataFrame(claims).to_csv(TABLES_DIR / "number_audit_enumeration_claims.csv", index=False)
        print(f"\nwrote {TABLES_DIR / 'number_audit_enumeration_claims.csv'}")


if __name__ == "__main__":
    main()
