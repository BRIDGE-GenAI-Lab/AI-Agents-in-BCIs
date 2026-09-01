"""TASK 21: repeated-attempt empirical replay of the assistive-action system.

Protocol frozen at docs/specs/2026-09-01-repeated-attempt-replay-protocol.md and
committed BEFORE this script produced a number.

MAKES NO API CALL AND SPENDS NOTHING. Every decision replayed here was already
recorded in output/intermediate/runs_natural/ by Task 20.

Run: UV_PROJECT_ENVIRONMENT=/private/tmp/nag_venv PYTHONPATH=code python3 \
     code/scripts/21_repeated_attempt_replay.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from nag.naturalistic import NATURAL_COMMANDS               # noqa: E402
from nag.replay import paired_differences, replay_panel     # noqa: E402
import nag.openrouter as _openrouter                         # noqa: E402


def _forbid_network_calls() -> None:
    """Make the zero-API-call guarantee real, not a proxy check.

    The plan's own draft guarded this with
    ``assert "nag.openrouter" not in sys.modules``. That check is broken by
    construction: `NATURAL_COMMANDS` lives in `nag.naturalistic`, which
    imports `nag.openrouter` at module load time (for `ParseFailure` and
    `extract_tool_calls`) purely to build episodes, and merely importing a
    module registers it in `sys.modules` whether or not it ever issues a
    request. The assertion would fail on line 1 of every run, including a
    run that truly spends $0.00. Importing costs nothing; only `chat` (the
    request loop) and `resolve_endpoint` (the provider-pinning GET) reach the
    network, so those two are the functions this script actually must never
    call. Patching them to raise makes that guarantee enforced by code
    instead of asserted by a check that can't pass.
    """
    def _refuse(name):
        def _raise(*args, **kwargs):
            raise RuntimeError(
                f"nag.openrouter.{name} called: repeated-attempt replay must make "
                "no API call and spends $0.00 by construction")
        return _raise
    _openrouter.chat = _refuse("chat")
    _openrouter.resolve_endpoint = _refuse("resolve_endpoint")


RUNS = REPO_ROOT / "output" / "intermediate" / "runs_natural"
TABLES = REPO_ROOT / "output" / "tables"
SPEC = REPO_ROOT / "docs" / "specs" / "2026-09-01-repeated-attempt-replay-protocol.md"

N_EPISODES = 200
MODELS = ("claude-sonnet-5", "gemini-3.7-flash", "gpt-5.6-luna",
          "glm-5.3-flash", "deepseek-v4-flash")
ENDPOINTS = ("p_success", "p_wrong", "p_wrong_tier3", "p_unresolved",
            "e_attempts", "success_per_100_attempts")

POLICY_FILES = {
    "gate:confidence": "natural_confidence_gate_canonical.parquet",
    "gate:lexical": "natural_confidence_gate_lexical.parquet",
}
for slug, model in [("anthropic_claude_sonnet_5", "claude-sonnet-5"),
                    ("google_gemini_3_7_flash", "gemini-3.7-flash"),
                    ("openai_gpt_5_6_luna", "gpt-5.6-luna"),
                    ("z_ai_glm_5_3_flash", "glm-5.3-flash"),
                    ("deepseek_deepseek_v4_flash", "deepseek-v4-flash")]:
    for arm, cell in [("none", "factorial_none_advisory_s0"),
                      ("advisory", "factorial_decoder_confidence_advisory_s0"),
                      ("enforced", "factorial_decoder_confidence_enforced_s0")]:
        POLICY_FILES[f"{model}:{arm}"] = f"{slug}__{cell}.parquet"


def _assert_protocol_is_frozen() -> None:
    """A protocol that can still be edited is not frozen.

    The whole value of committing it first is lost if it carries uncommitted
    changes while producing results, so this refuses rather than warns.
    """
    rel = SPEC.relative_to(REPO_ROOT)
    diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", str(rel)],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    if diff.stdout.strip():
        raise SystemExit(f"{rel} has uncommitted changes; commit it before running")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        raise SystemExit(f"{rel} is not tracked by git; commit it before running")


def load() -> dict:
    frames = {}
    for name, fname in POLICY_FILES.items():
        f = RUNS / fname
        # "._*" are real files on this exFAT volume and are not parquet.
        if f.name.startswith("._") or not f.exists():
            raise SystemExit(f"missing policy file {fname}")
        d = pd.read_parquet(f)
        if len(d) != N_EPISODES:
            raise SystemExit(f"{fname}: {len(d)} rows, expected {N_EPISODES}")
        if set(d["assigned_command"]) != set(NATURAL_COMMANDS):
            raise SystemExit(f"{fname}: command set does not match the frozen nine")
        frames[name] = d
    return frames


def main() -> int:
    _assert_protocol_is_frozen()
    _forbid_network_calls()

    frames = load()
    spent = sum(float(d["cost_usd"].sum()) for d in frames.values())
    print(f"replaying {len(frames)} policies over {N_EPISODES} episodes, "
          f"{frames['gate:confidence']['participant_id'].nunique()} participants")
    print(f"cost of this analysis: $0.00 (the ${spent:.2f} below was already spent by Task 20)")

    primary = replay_panel(frames, commands=NATURAL_COMMANDS,
                           n_boot=2000, seed=20260901, with_replacement=False)
    primary.insert(1, "draw", "without_replacement")
    sens = replay_panel(frames, commands=NATURAL_COMMANDS,
                        n_boot=2000, seed=20260901, with_replacement=True)
    sens.insert(1, "draw", "with_replacement")
    out = pd.concat([primary, sens], ignore_index=True)
    out.to_csv(TABLES / "repeated_attempt_replay.csv", index=False)

    # Contrasts, paired within model. The headline is enforcement against
    # judgement; the lexical resolver is a reference only, because it is given
    # the nine canonical command strings and the models are not. Each
    # difference gets its OWN joint-draw bootstrap interval via
    # `paired_differences`, not two marginal intervals combined after the
    # fact -- and all six endpoints for one contrast come from a SINGLE
    # bootstrap pass (one call per pair, not one call per endpoint), because
    # `_endpoints_for` already returns every endpoint on every replicate.
    rows = []
    for model in MODELS:
        for label, a, b in [
            ("enforced_minus_advisory", f"{model}:enforced", f"{model}:advisory"),
            ("advisory_minus_none", f"{model}:advisory", f"{model}:none"),
            ("gate_minus_advisory", "gate:confidence", f"{model}:advisory"),
        ]:
            results = paired_differences(frames, a, b, ENDPOINTS, commands=NATURAL_COMMANDS,
                                         n_boot=2000, seed=20260901, with_replacement=False)
            for ep in ENDPOINTS:
                r = results[ep]
                rows.append({"model": model, "contrast": label, "endpoint": ep,
                             "a": r["a"], "b": r["b"], "difference": r["difference"],
                             "lo": r["lo"], "hi": r["hi"],
                             "n_boot_dropped": r["n_boot_dropped"]})
    pd.DataFrame(rows).to_csv(TABLES / "repeated_attempt_contrasts.csv", index=False)

    print(f"wrote {TABLES / 'repeated_attempt_replay.csv'}")
    print(f"wrote {TABLES / 'repeated_attempt_contrasts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
