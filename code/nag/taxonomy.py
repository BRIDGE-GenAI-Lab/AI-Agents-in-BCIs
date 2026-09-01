"""Frozen dense codebook: decoded string -> entailed action.

This module defines a channel code, not a claim about participant intent.
Every string over the 36-character speller alphabet maps to exactly one of
nine actions via a frozen salted hash. The mapping is deliberately dense --
a real low-bandwidth speller agent has no spare code space for "invalid"
codes, so a single-character decode error almost always still decodes to
some OTHER valid action rather than to nothing. That silent substitution is
the phenomenon this study measures. `entail()` reports only what a given
string entails under this frozen code; it never asserts what the person who
produced the underlying neural signal meant to communicate.

Consequence tiers (recorded here, never surfaced in any agent-facing prompt
text -- see tests/test_taxonomy.py::test_tier_label_never_leaks_into_prompt_surface):
  tier 1 -- reversible and private
  tier 2 -- reversible but observed by others
  tier 3 -- irreversible or externally consequential
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

_PATH = Path(__file__).with_name("frozen_mapping.json")


@dataclass(frozen=True)
class Action:
    """An entailed tool call. Equality is (name, args) only -- `tier` is
    excluded from comparison. `agent._as_action()` (Task 4) builds an Action
    from an executed tool call and has no way to know its consequence tier,
    so comparing the whole dataclass would score every correct action as
    unfaithful and silently null out the study's results.
    """
    name: str
    args: tuple
    tier: int = field(compare=False)


def load_mapping(path: Path = _PATH) -> dict:
    """Load the frozen codebook parameters (salt, action order, tiers)."""
    return json.loads(path.read_text())


_PARAMS = load_mapping()
SALT: str = _PARAMS["salt"]
ACTIONS: list[str] = _PARAMS["actions"]
TIERS: dict[str, int] = _PARAMS["tiers"]
_ALPHABET: str = _PARAMS["alphabet"]
_STRING_LENGTH: int = _PARAMS["string_length"]
_ALPHABET_SET = frozenset(_ALPHABET)

assert len(ACTIONS) == 9
assert set(TIERS) == set(ACTIONS)


def mapping_digest(path: Path = _PATH) -> str:
    """sha256 hex digest of the frozen mapping file. Recorded in every run
    manifest so a downstream run can pin which codebook produced it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entail(s: str) -> Action | None:
    """The action a given string entails under the frozen dense codebook.

    Returns None only for malformed input (wrong length, or characters
    outside the 36-character speller alphabet) -- never for "unmapped",
    because the codebook is dense over every string of the configured
    length.
    """
    if not isinstance(s, str) or len(s) != _STRING_LENGTH:
        return None
    if not _ALPHABET_SET.issuperset(s):
        return None
    index = int(hashlib.sha256((SALT + s).encode()).hexdigest(), 16) % len(ACTIONS)
    name = ACTIONS[index]
    return Action(name=name, args=(), tier=TIERS[name])
