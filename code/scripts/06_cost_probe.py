"""Measure REAL per-episode cost by running a handful of episodes end to end.

Replaces the modelled cost envelope with measurement. Every OpenRouter response
carries `usage.cost` in exact USD plus cache hit/write token counts, so the
budget question becomes arithmetic on observed values instead of assumptions
about tokens-per-call and calls-per-episode.

Writes output/tables/cost_probe.json. Never prints or stores the API key.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nag.agent import run_episode
from nag.design import enumerate_cells
from nag.episodes import build_episodes
from nag.openrouter import chat, resolve_endpoint
from nag.prompts import build_system
from nag.tools import TOOL_SCHEMAS

TRIALS = ("../study_bigp3_als_calibration/output/intermediate/"
          "online_trials_all20.csv")
SCORES = "output/intermediate/selection_scores.parquet"
OUT = Path("output/tables/cost_probe.json")

# Span the price range so the blend can be checked, not assumed.
MODELS = [
    "meta-llama/llama-3.3-70b-instruct",   # open-weight, cheap end
    "qwen/qwen-2.5-72b-instruct",          # open-weight alternate
    "openai/gpt-4o-mini",                  # commercial, mid
]
N_EPISODES = 3
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 1.0


class PinnedClient:
    """Adapts nag.openrouter.chat to the run_episode client contract.

    `provider` here MUST be the resolved endpoint's TAG
    (`resolve_endpoint(...)["tag"]`), never its `provider_name` (Ruling 29:
    a display name like "Amazon Bedrock" is not a routable pin and 404s
    against the live API). `provider_name` is accepted separately, purely
    to thread the human-readable display name into the provenance record
    for reporting -- it is never itself sent as the pin.
    """

    def __init__(self, model, provider, api_key, quantization, context_length,
                 supports_tools=True, provider_name=None):
        self.model = model
        self.provider = provider
        self.provider_name = provider_name
        self.api_key = api_key
        self.quantization = quantization
        self.context_length = context_length
        self.supports_tools = supports_tools
        self.records: list[dict] = []

    def chat(self, messages, tools):
        os.environ["OPENROUTER_API_KEY"] = self.api_key
        return chat(
            model=self.model, messages=messages, tools=tools,
            provider=self.provider, temperature=TEMPERATURE, top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            quantization=self.quantization, context_length=self.context_length,
            supports_tools=self.supports_tools, provider_name=self.provider_name,
            record_hook=self.records.append,
        )


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    episodes = build_episodes(pd.read_csv(TRIALS), length=3, als_only=True)
    scored = set(pd.read_parquet(SCORES)["relative_path"])
    episodes = episodes[episodes.relative_path.isin(scored)] if "relative_path" in episodes else episodes
    sample = episodes.head(N_EPISODES).to_dict("records")

    # A representative advisory cell: the prompt carries an uncertainty value,
    # so token counts reflect the fuller end of the prompt distribution.
    cell = next(c for c in enumerate_cells()
                if c.uncertainty_source == "decoder_confidence"
                and c.control_mechanism == "advisory")
    system = build_system(cell, confidence=0.62)

    results = []
    for model in MODELS:
        try:
            ep = resolve_endpoint(model, key)
            # `tag` is the routable pin (Ruling 29); `provider_name` is kept
            # only as the human-readable display name for the report below
            # and for the provenance record -- never sent as the pin itself.
            tag = ep["tag"] if isinstance(ep, dict) else ep.tag
            prov = ep["provider_name"] if isinstance(ep, dict) else ep.provider_name
            quant = ep["quantization"] if isinstance(ep, dict) else ep.quantization
            ctx = ep["context_length"] if isinstance(ep, dict) else ep.context_length
            # resolve_endpoint's `candidates` list (every endpoint OpenRouter
            # reported, ~8KB for a busy model) is deliberately NOT carried
            # into `results` below -- only the chosen endpoint plus a count,
            # so this script's output stays a small, readable summary rather
            # than ballooning with the full per-model endpoint dump.
            n_candidates = len(ep["candidates"]) if isinstance(ep, dict) else len(ep.candidates)
            supports_tools = ep["supports_tools"] if isinstance(ep, dict) else ep.supports_tools
        except Exception as e:  # endpoint resolution is a hard failure, report it
            # Covers NoToolCapableEndpoint too: a model with no tool-capable
            # endpoint at all fails loudly here, at resolution time, rather
            # than silently dropping episodes with a 404 mid-run.
            results.append({"model": model, "error": f"{type(e).__name__}: {e}"})
            continue

        client = PinnedClient(model, tag, key, quant, ctx, supports_tools, provider_name=prov)
        per_ep = []
        for episode in sample:
            before = len(client.records)
            try:
                rec = run_episode(cell=cell, episode=episode, confidence=0.62,
                                  client=client, system=system)
                calls = client.records[before:]
                per_ep.append({
                    "episode_id": episode["episode_id"],
                    "n_calls": len(calls),
                    "n_turns": rec.n_turns,
                    "covered": rec.covered,
                    "faithful": rec.faithful,
                    "parse_failed": rec.parse_failed,
                    "cost_usd": sum(c.get("cost") or 0 for c in calls),
                    "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
                    "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
                    "cached_tokens": sum(c.get("cached_tokens") or 0 for c in calls),
                })
            except Exception as e:
                per_ep.append({"episode_id": episode["episode_id"],
                               "error": f"{type(e).__name__}: {str(e)[:200]}"})

        ok = [e for e in per_ep if "error" not in e]
        results.append({
            "model": model, "provider": prov, "tag": tag, "quantization": quant,
            "context_length": ctx, "supports_tools": supports_tools,
            "n_candidate_endpoints": n_candidates, "episodes": per_ep,
            "n_ok": len(ok),
            "mean_cost_usd": (sum(e["cost_usd"] for e in ok) / len(ok)) if ok else None,
            "mean_calls": (sum(e["n_calls"] for e in ok) / len(ok)) if ok else None,
            "mean_prompt_tokens": (sum(e["prompt_tokens"] for e in ok) / len(ok)) if ok else None,
            "mean_completion_tokens": (sum(e["completion_tokens"] for e in ok) / len(ok)) if ok else None,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"system_prompt_chars": len(system), "n_tools": len(TOOL_SCHEMAS),
               "max_tokens": MAX_TOKENS, "results": results}
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
