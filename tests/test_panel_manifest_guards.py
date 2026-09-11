"""The run manifest's self-consistency about WHICH panel and WHICH ceiling.

Two facts about this manifest are load-bearing for a study that has already
been submitted, and neither is checked by anything that runs the experiments:

1. `05b_panel.py` holds the budget ceiling as a module constant while the human
   raises it in the manifest. Twice now the constant has been left behind (at
   100.0 while the manifest said 115.0, then at 115.0 while it said 120.0), so a
   re-run of that script would have silently reverted a raise the runners were
   already spending against.

2. The model panel was expanded from five models to ten AFTER the manuscript's
   numbers were produced, and the five frozen runners read the panel LIVE. A
   reproducer who ran them today would get a ten-model run and no warning.

Both are recorded rather than remembered, and this file is what keeps them so.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PANEL_SCRIPT = REPO / "code" / "scripts" / "05b_panel.py"
MANIFEST = REPO / "output" / "tables" / "run_manifest.json"

PUBLISHED_PANEL = [
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.7-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash",
]


def _load_panel_script():
    spec = importlib.util.spec_from_file_location("_t05b", PANEL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_panel_scripts_budget_constant_matches_the_ceiling_the_manifest_carries():
    """The drift itself, caught at the source. `05b_panel.py` writes its constant
    into the manifest, so any gap between the two is a raise waiting to be
    reverted by the next person who runs the script for an unrelated reason."""
    mod = _load_panel_script()
    manifest = json.loads(MANIFEST.read_text())
    assert mod.BUDGET_USD == manifest["budget_usd"], (
        f"05b_panel.BUDGET_USD is {mod.BUDGET_USD} but the manifest carries "
        f"{manifest['budget_usd']} -- running the script would move the human's ceiling"
    )


def test_the_panel_script_refuses_to_lower_a_ceiling_the_manifest_already_carries(tmp_path):
    """Run against the break it exists to catch, not against good data: a
    manifest whose ceiling is ABOVE the script's constant, which is exactly the
    state both past drifts produced. The script must refuse and write nothing --
    a guard that only ever sees a matching pair would pass either way."""
    mod = _load_panel_script()
    manifest = json.loads(MANIFEST.read_text())
    manifest["budget_usd"] = mod.BUDGET_USD + 10.0
    manifest["budget_note"] = "the human's audit trail, which must survive a refusal"
    sandbox = tmp_path / "run_manifest.json"
    sandbox.write_text(json.dumps(manifest, indent=2))
    before = sandbox.read_text()

    mod.MANIFEST = sandbox
    with pytest.raises(SystemExit, match="REFUSING TO WRITE"):
        mod.main()
    assert sandbox.read_text() == before, "the script refused but wrote anyway"


def test_the_manifest_names_the_five_models_that_produced_the_published_numbers():
    """`model_panel` now holds ten models and the five runners that produced the
    manuscript read it live, so without this block nothing in the repo says which
    five ran. The block also has to keep saying how to reproduce them, since the
    answer is a flag rather than the default."""
    manifest = json.loads(MANIFEST.read_text())
    block = manifest["model_panel_at_principal_run"]
    assert block["models"] == PUBLISHED_PANEL
    assert block["n_models"] == 5
    assert "--models" in block["reproduction_note"]

    current = [m["slug"] for m in manifest["model_panel"]["models"]]
    assert current[:5] == PUBLISHED_PANEL, "the published five must remain identifiable in the panel"
    assert manifest["model_panel"]["n_models"] == len(current)


def test_the_never_executed_full_factorial_projection_is_labelled_as_such():
    """`main_run.episode_runs.total` reads 32000, a design nobody ran: the block
    is a projection that scales with the current panel size. Left unlabelled
    beside `principal_run`, it reads as a second, larger experiment."""
    manifest = json.loads(MANIFEST.read_text())
    status = manifest["main_run"]["status"]
    assert status.startswith("HISTORICAL / SUPERSEDED")
    assert "principal_run" in status and "model_panel_at_principal_run" in status
    assert manifest["main_run"]["episode_runs"]["total"] == (
        manifest["model_panel"]["n_models"] * manifest["main_run"]["llm_cells"]
        * manifest["main_run"]["n_episodes_per_cell"]
    ), "the projection no longer even matches the panel it is projected over"
