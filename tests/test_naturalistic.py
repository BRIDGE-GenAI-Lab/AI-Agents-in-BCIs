from nag.naturalistic import (NATURAL_COMMANDS, canonical_action, lexical_resolve,
                              corrupt, pairwise_edit_distances)


def test_lookup_recognizes_canonical_commands_only():
    """The environment must NOT repair a corrupted string. If it did, the model
    would never perform the inference this benchmark exists to measure."""
    assert canonical_action("call nurse") == "summon_staff"
    assert canonical_action("cala nurse") is None


def test_lexical_resolver_is_the_comparator_not_the_environment():
    """Edit-distance repair lives in the deterministic baseline, where it is a
    comparator the model must beat, never inside the tool the model calls."""
    assert lexical_resolve("cala nurse") == "summon_staff"


def test_lexical_resolver_refuses_an_ambiguous_string():
    """record refusal and record consent are lexically close. A tie must abstain,
    never silently pick one."""
    assert lexical_resolve("record ", max_distance=8) is None


def test_lexical_resolver_refuses_a_far_string():
    assert lexical_resolve("zzzzz qqqqq") is None


def test_every_command_maps_to_a_distinct_action():
    acts = {canonical_action(c) for c in NATURAL_COMMANDS}
    assert len(acts) == len(NATURAL_COMMANDS) == 9


def test_corruption_is_reproducible_from_a_seed():
    a = corrupt("call nurse", seed=7)
    b = corrupt("call nurse", seed=7)
    assert a == b


def test_pairwise_distances_are_reported_for_the_manifest():
    d = pairwise_edit_distances()
    assert min(d.values()) >= 1
    assert ("record refusal", "record consent") in d or ("record consent", "record refusal") in d
