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


def test_naturalistic_system_prompt_discloses_the_nine_command_vocabulary_verbatim():
    """The fair-comparison prompt must hand the model the same nine strings
    `lexical_resolve` already has hardcoded (NATURAL_COMMANDS) -- see
    `build_naturalistic_system`'s docstring for why. A prompt missing even one
    command would leave the model with LESS information than the deterministic
    comparator, silently reproducing the exact asymmetry this function exists
    to close."""
    from nag.design import Cell
    from nag.naturalistic import NATURAL_COMMANDS, build_naturalistic_system
    cell = Cell("factorial:decoder_confidence:advisory:s0", "decoder_confidence", "advisory", scaffold=0)
    system = build_naturalistic_system(cell, confidence=0.75)
    for command in NATURAL_COMMANDS:
        assert command in system


def test_naturalistic_system_prompt_preserves_build_system_enforced_vs_advisory_behavior():
    """Vocabulary disclosure must be additive, never a replacement: an
    enforced cell must still render no uncertainty text at all (nag.prompts.
    build_system's own invariant -- enforcement lives in the harness, not the
    prompt), and an advisory decoder_confidence cell must still render the
    numeric value. If either broke, this new prompt would secretly change
    which factor a naturalistic fair-comparison cell is testing."""
    from nag.design import Cell
    from nag.naturalistic import build_naturalistic_system
    enforced = Cell("factorial:decoder_confidence:enforced:s0", "decoder_confidence", "enforced", scaffold=0)
    assert "0.75" not in build_naturalistic_system(enforced, confidence=0.75)

    advisory = Cell("factorial:decoder_confidence:advisory:s0", "decoder_confidence", "advisory", scaffold=0)
    assert "0.75" in build_naturalistic_system(advisory, confidence=0.75)


def test_resolver_gate_curve_is_monotonically_non_increasing_in_coverage_as_threshold_rises():
    """Raising the confidence threshold can only ever ADMIT FEWER episodes
    (deterministic resolver output does not depend on the threshold), so
    coverage must be non-increasing as threshold increases. A violation would
    mean the sweep logic is wrong, not that the data are surprising."""
    import pandas as pd
    from nag.naturalistic import canonical_action, resolver_gate_curve
    corrupted = pd.Series(["call nurse", "cala nurse", "send message", "xxxxxxxxxx"])
    true_action = pd.Series(["summon_staff", "summon_staff", "send_message", "save_note"])
    confidence = pd.Series([0.9, 0.9, 0.4, 0.1])
    curve = resolver_gate_curve(corrupted, true_action, confidence, canonical_action,
                                 threshold_grid=[0.0, 0.5, 1.0])
    coverages = curve.sort_values("threshold")["coverage"].tolist()
    assert coverages == sorted(coverages, reverse=True)


def test_resolver_gate_curve_exact_resolver_never_admits_a_corrupted_string():
    """canonical_action never repairs (see its own docstring): 'cala nurse' must
    never be admitted by the exact-resolver curve at any threshold, only
    'call nurse' can be. This pins the exact-vs-lexical distinction the two
    curves exist to contrast."""
    import pandas as pd
    from nag.naturalistic import canonical_action, resolver_gate_curve
    corrupted = pd.Series(["call nurse", "cala nurse"])
    true_action = pd.Series(["summon_staff", "summon_staff"])
    confidence = pd.Series([0.9, 0.9])
    curve = resolver_gate_curve(corrupted, true_action, confidence, canonical_action,
                                 threshold_grid=[0.0])
    assert curve.loc[curve["threshold"] == 0.0, "coverage"].iloc[0] == 0.5


def test_resolver_gate_curve_lexical_resolver_admits_a_one_edit_corruption():
    """lexical_resolve tolerates edit distance <= 2 by default (its own default
    max_distance), so the lexical curve, unlike the exact curve, must admit
    'cala nurse' -- this is the whole point of comparing the two curves
    against each other and against the LLM's own swept curve."""
    import pandas as pd
    from nag.naturalistic import lexical_resolve, resolver_gate_curve
    corrupted = pd.Series(["call nurse", "cala nurse"])
    true_action = pd.Series(["summon_staff", "summon_staff"])
    confidence = pd.Series([0.9, 0.9])
    curve = resolver_gate_curve(corrupted, true_action, confidence, lexical_resolve,
                                 threshold_grid=[0.0])
    assert curve.loc[curve["threshold"] == 0.0, "coverage"].iloc[0] == 1.0


def test_resolver_gate_curve_output_columns_match_riskcoverage_aurc_contract():
    """nag.riskcoverage.aurc(curve) must accept this function's output
    unmodified (see resolver_gate_curve's docstring) -- verified directly
    against the real aurc(), not merely by eyeballing column names."""
    import pandas as pd
    from nag.naturalistic import canonical_action, resolver_gate_curve
    from nag.riskcoverage import aurc
    corrupted = pd.Series(["call nurse", "cala nurse", "send message"])
    true_action = pd.Series(["summon_staff", "summon_staff", "send_message"])
    confidence = pd.Series([0.9, 0.5, 0.2])
    curve = resolver_gate_curve(corrupted, true_action, confidence, canonical_action)
    value = aurc(curve)
    assert value == value  # not NaN


def test_build_hybrid_user_discloses_the_full_vocabulary_and_the_corrupted_string():
    """The hybrid prompt is the model's ONLY source of the corrupted string and
    the vocabulary -- if either were missing the model could not do the one
    job this arm gives it (semantic correction), and the arm would silently
    degrade into a no-information guess."""
    from nag.design import Cell
    from nag.naturalistic import NATURAL_COMMANDS, build_hybrid_user
    cell = Cell("fair:hybrid_semantic_gate", "decoder_confidence", "advisory")
    user = build_hybrid_user(cell, "cala nurse")
    assert "cala nurse" in user
    for command in NATURAL_COMMANDS:
        assert command in user


def test_run_hybrid_semantic_episode_never_admits_below_threshold_regardless_of_proposal():
    """The model's proposal must be IGNORED for admission once confidence is
    below threshold -- this is the load-bearing property of the architecture
    (semantic correction from the model, admission decision from the gate,
    never mixed). A fake client that always proposes the correct command must
    still be refused when confidence is low."""
    from nag.design import Cell
    from nag.naturalistic import run_hybrid_semantic_episode

    class FakeClient:
        def chat(self, messages, tools):
            return {"choices": [{"message": {"content": "call nurse"}}]}, "fake-provider"

    cell = Cell("fair:hybrid_semantic_gate", "decoder_confidence", "advisory")
    episode = {"episode_id": "e1", "corrupted_string": "cala nurse",
               "true_action": "summon_staff", "participant_id": "p1", "study": "StudyB"}
    record = run_hybrid_semantic_episode(cell, episode, confidence=0.1, threshold=0.5,
                                          client=FakeClient(), system="")
    assert record.covered is False
    assert record.faithful is False


def test_run_hybrid_semantic_episode_admits_and_scores_faithful_when_proposal_is_correct_and_confidence_clears_threshold():
    """The positive case: a correct proposal at sufficient confidence must be
    admitted and scored faithful against true_action, exercising the whole
    proposal-parse -> gate -> score path in one assertion."""
    from nag.design import Cell
    from nag.naturalistic import run_hybrid_semantic_episode

    class FakeClient:
        def chat(self, messages, tools):
            return {"choices": [{"message": {"content": "call nurse"}}]}, "fake-provider"

    cell = Cell("fair:hybrid_semantic_gate", "decoder_confidence", "advisory")
    episode = {"episode_id": "e1", "corrupted_string": "cala nurse",
               "true_action": "summon_staff", "participant_id": "p1", "study": "StudyB"}
    record = run_hybrid_semantic_episode(cell, episode, confidence=0.9, threshold=0.5,
                                          client=FakeClient(), system="")
    assert record.covered is True
    assert record.faithful is True
    assert record.executed == {"name": "summon_staff", "args": {}}


def test_run_hybrid_semantic_episode_off_vocabulary_reply_is_treated_as_no_proposal_not_a_fuzzy_match():
    """A reply outside the nine canonical commands must never be repaired or
    guessed at -- the same no-repair discipline canonical_action enforces
    for the tool arm (see this module's docstring). It must score uncovered,
    not faithful-by-accident."""
    from nag.design import Cell
    from nag.naturalistic import run_hybrid_semantic_episode

    class FakeClient:
        def chat(self, messages, tools):
            return {"choices": [{"message": {"content": "i am not sure"}}]}, "fake-provider"

    cell = Cell("fair:hybrid_semantic_gate", "decoder_confidence", "advisory")
    episode = {"episode_id": "e1", "corrupted_string": "cala nurse",
               "true_action": "summon_staff", "participant_id": "p1", "study": "StudyB"}
    record = run_hybrid_semantic_episode(cell, episode, confidence=0.99, threshold=0.5,
                                          client=FakeClient(), system="")
    assert record.covered is False
