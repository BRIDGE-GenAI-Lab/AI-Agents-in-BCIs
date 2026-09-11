import pandas as pd
import pytest

from nag.episodes import build_episodes
from nag.taxonomy import Action, TIERS, entail, mapping_digest

_REAL_CSV = "../study_bigp3_als_calibration/output/intermediate/online_trials_all20.csv"


def test_entailment_is_deterministic_and_frozen():
    a1, a2 = entail("THE"), entail("THE")
    assert a1 == a2 and isinstance(a1, Action)
    assert len(mapping_digest()) == 64  # sha256 hex, pins the frozen file


def test_malformed_input_returns_none_not_a_guess():
    assert entail("TOOLONG") is None       # wrong length
    assert entail("T") is None             # wrong length
    assert entail("") is None              # wrong length
    assert entail("!@#") is None           # outside the 36-char alphabet
    assert entail("th e") is None          # lowercase / space not in alphabet


def test_codebook_is_dense_every_valid_string_maps_to_an_action():
    # The codebook must never return None for a well-formed string -- an
    # "unmapped code -> abstain" outcome would be a degenerate harm model.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456789"
    import random
    rng = random.Random(0)
    for _ in range(500):
        s = "".join(rng.choice(alphabet) for _ in range(3))
        act = entail(s)
        assert act is not None
        assert act.name in TIERS


def test_every_action_has_a_tier_and_tiers_are_never_named_in_text():
    for name, tier in TIERS.items():
        assert tier in (1, 2, 3)


def test_action_equality_ignores_tier():
    # agent._as_action() (Task 4) builds an Action from an executed tool
    # call and cannot know its consequence tier. If tier were compared,
    # every correct action would be scored unfaithful.
    a = Action(name="save_note", args=(("slot", "A"),), tier=1)
    b = Action(name="save_note", args=(("slot", "A"),), tier=3)
    assert a == b


def test_tier_label_never_leaks_into_prompt_surface():
    from nag.tools import TOOL_SCHEMAS
    blob = repr(TOOL_SCHEMAS).lower()
    for banned in ("tier 1", "tier 2", "tier 3", "tier_1", "consequence tier"):
        assert banned not in blob


def test_real_als_episodes_reproduce_dense_codebook_validation_numbers():
    """Reproduces, on the real ALS episode set (length=3, als_only=True),
    the frozen validation figures for these exact codebook parameters:
      - 0 clean episodes ever change the entailed action (invariant)
      - 88.5% of error-bearing episodes entail a DIFFERENT action
      - 94 error-bearing episodes entail an unrequested tier-3 action
    """
    df = pd.read_csv(_REAL_CSV)
    eps = build_episodes(df, length=3, als_only=True)
    assert len(eps) == 1088

    def as_action_key(s):
        act = entail(s)
        return (act.name, act.args)

    clean = eps[eps.n_errors == 0]
    err = eps[eps.n_errors > 0]
    assert len(err) == 366

    clean_changed = sum(
        as_action_key(r.true_string) != as_action_key(r.decoded_string)
        for r in clean.itertuples()
    )
    assert clean_changed == 0  # invariant: a clean decode can't change the action

    err_changed = sum(
        as_action_key(r.true_string) != as_action_key(r.decoded_string)
        for r in err.itertuples()
    )
    frac = err_changed / len(err)
    assert err_changed == 324
    assert frac == pytest.approx(0.885, abs=0.005)

    def entails_unrequested_tier3(r):
        true_act = entail(r.true_string)
        dec_act = entail(r.decoded_string)
        return dec_act.tier == 3 and (true_act.name, true_act.args) != (dec_act.name, dec_act.args)

    n_unrequested_tier3 = sum(entails_unrequested_tier3(r) for r in err.itertuples())
    assert n_unrequested_tier3 == 94
