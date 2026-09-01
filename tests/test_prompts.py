from nag.design import Cell
from nag.prompts import CAUTION_WORDINGS, SCAFFOLDS, build_system, prompts_digest

_BANNED_TIER_WORDS = ("tier", "tier 1", "tier 2", "tier 3", "consequence tier", "irreversible")
_BANNED_SAFETY_WORDS = ("careful", "confirm", "uncertain", "noisy", "double-check", "verify")


def test_instrument_is_frozen_and_complete():
    assert len(CAUTION_WORDINGS) == 12 and len(set(CAUTION_WORDINGS)) == 12
    assert len(SCAFFOLDS) == 3 and len(set(SCAFFOLDS)) == 3
    assert len(prompts_digest()) == 64
    assert prompts_digest() == prompts_digest()  # deterministic, not re-salted per call


def test_no_tier_label_leaks_into_any_prompt_text():
    blob = " ".join(CAUTION_WORDINGS + SCAFFOLDS).lower()
    for banned in _BANNED_TIER_WORDS:
        assert banned not in blob, banned


def test_scaffold_paraphrases_carry_no_safety_content():
    """Scaffold is a nuisance factor; it must not smuggle in the exposure."""
    for s in SCAFFOLDS:
        low = s.lower()
        for leak in _BANNED_SAFETY_WORDS:
            assert leak not in low, (s, leak)


def test_actual_posterior_string_never_appears():
    blob = " ".join(CAUTION_WORDINGS + SCAFFOLDS)
    assert "actual posterior" not in blob.lower()
    for cell in _sample_cells():
        assert "actual posterior" not in build_system(cell, confidence=0.5).lower()


def test_uncertainty_value_appears_only_for_advisory_decoder_confidence_cell():
    none_cell = Cell("factorial:none:advisory:s0", "none", "advisory")
    dec_cell = Cell("factorial:decoder_confidence:advisory:s0", "decoder_confidence", "advisory")
    assert "0.73" not in build_system(none_cell, confidence=0.73)
    assert "0.73" in build_system(dec_cell, confidence=0.73)


def test_confidence_value_never_appears_for_an_enforced_cell():
    dec_enforced = Cell("factorial:decoder_confidence:enforced:s0", "decoder_confidence", "enforced")
    oracle = Cell("oracle", "oracle", "enforced")
    assert "0.73" not in build_system(dec_enforced, confidence=0.73)
    assert "0.73" not in build_system(oracle, confidence=0.73)
    # enforced renders NO uncertainty text at all, not just no number
    assert build_system(dec_enforced, confidence=0.73) == build_system(dec_enforced, confidence=None)


def test_confidence_value_never_appears_for_a_none_source():
    none_advisory = Cell("factorial:none:advisory:s0", "none", "advisory")
    assert "0.73" not in build_system(none_advisory, confidence=0.73)


def test_self_confidence_gets_a_self_report_instruction_not_a_number():
    self_conf = Cell("factorial:self_confidence:advisory:s0", "self_confidence", "advisory")
    out = build_system(self_conf, confidence=0.73)
    assert "0.73" not in out
    assert "confidence" in out.lower()


def test_caution_wording_only_appears_for_the_caution_family():
    baseline = Cell("factorial:none:advisory:s0", "none", "advisory", scaffold=0, wording=0)
    caution0 = Cell("caution:w0", "none", "advisory", scaffold=0, wording=0)
    # identical on every field but `name` -- name is the only signal that
    # distinguishes the caution family from the matching factorial baseline
    assert baseline.uncertainty_source == caution0.uncertainty_source
    assert baseline.control_mechanism == caution0.control_mechanism
    assert baseline.scaffold == caution0.scaffold
    assert baseline.wording == caution0.wording
    assert CAUTION_WORDINGS[0] not in build_system(baseline)
    assert CAUTION_WORDINGS[0] in build_system(caution0)
    assert build_system(baseline) != build_system(caution0)


def test_caution_wording_index_is_selected_by_cell_wording():
    for w in range(12):
        cell = Cell(f"caution:w{w}", "none", "advisory", wording=w)
        out = build_system(cell)
        assert CAUTION_WORDINGS[w] in out
        for other in range(12):
            if other != w:
                assert CAUTION_WORDINGS[other] not in out


def test_scaffold_is_selected_by_cell_scaffold():
    for k in range(3):
        cell = Cell(f"factorial:none:advisory:s{k}", "none", "advisory", scaffold=k)
        out = build_system(cell)
        assert out.startswith(SCAFFOLDS[k])


def test_non_caution_cells_never_render_a_caution_wording():
    for cell in (
        Cell("singleshot", "none", "advisory"),
        Cell("nonllm_gate", "decoder_confidence", "enforced", uses_llm=False),
        Cell("random_gate", "none", "enforced", uses_llm=False),
        Cell("oracle", "oracle", "enforced"),
    ):
        out = build_system(cell, confidence=0.5)
        for wording in CAUTION_WORDINGS:
            assert wording not in out


def _sample_cells():
    return [
        Cell("factorial:none:advisory:s0", "none", "advisory"),
        Cell("factorial:decoder_confidence:advisory:s1", "decoder_confidence", "advisory", scaffold=1),
        Cell("factorial:self_confidence:advisory:s2", "self_confidence", "advisory", scaffold=2),
        Cell("factorial:decoder_confidence:enforced:s0", "decoder_confidence", "enforced"),
        Cell("caution:w5", "none", "advisory", wording=5),
        Cell("oracle", "oracle", "enforced"),
        Cell("singleshot", "none", "advisory"),
    ]
