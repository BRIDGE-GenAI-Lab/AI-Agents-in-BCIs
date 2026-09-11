import json
import os
import random

import pytest
import requests

from nag import openrouter
from nag.openrouter import (
    NoToolCapableEndpoint,
    ParseFailure,
    ProviderMismatch,
    assert_provider,
    build_payload,
    chat,
    extract_tool_call,
    extract_tool_calls,
    record_request,
    resolve_endpoint,
)


# ---- build_payload -----------------------------------------------------

def test_payload_always_pins_provider_and_sets_sampling_explicitly():
    p = build_payload("x/y", [{"role": "user", "content": "hi"}], tools=[],
                       provider="together", temperature=0.7, top_p=1.0, max_tokens=512)
    assert p["provider"] == {"order": ["together"], "allow_fallbacks": False}
    for k in ("temperature", "top_p", "max_tokens"):
        assert k in p  # OpenRouter omits absent params upstream


def test_payload_never_sends_a_quantizations_filter():
    # A `quantizations` filter 404s against the live API for providers that
    # serve a different quantization of the same model slug (e.g. DeepInfra
    # fp8 vs Novita bf16 for the same llama-3.3-70b slug). Quantization is
    # discovered per-endpoint (resolve_endpoint), never imposed as a filter.
    p = build_payload("x/y", [], tools=[], provider="deepinfra",
                       temperature=0.0, top_p=0.9, max_tokens=64)
    assert "quantizations" not in p["provider"]


def test_payload_pin_reflects_whichever_provider_is_passed():
    p = build_payload("x/y", [], tools=[], provider="deepinfra",
                       temperature=0.0, top_p=0.9, max_tokens=64)
    assert p["provider"]["order"] == ["deepinfra"]
    assert p["provider"]["allow_fallbacks"] is False
    assert p["temperature"] == 0.0
    assert p["top_p"] == 0.9
    assert p["max_tokens"] == 64


# ---- assert_provider -----------------------------------------------------
# Verification reads the response BODY's top-level "provider" field, not a
# response header. Live probing found OpenRouter's documented
# "openrouter-selected-provider" header does not exist in practice.

def test_assert_provider_returns_served_provider_when_it_matches_pin():
    assert assert_provider({"provider": "together"}, "together") == "together"


def test_assert_provider_is_case_and_space_insensitive():
    # Pins are lowercase slugs ("deepinfra"); the body reports display names
    # ("DeepInfra", "Azure", "OpenAI").
    assert assert_provider({"provider": "DeepInfra"}, "deepinfra") == "DeepInfra"
    assert assert_provider({"provider": "Deep Infra"}, "deepinfra") == "Deep Infra"


def test_assert_provider_is_hyphen_insensitive_for_kebab_case_tag_portions():
    # Ruling 29: tag provider portions are kebab-case ("amazon-bedrock",
    # "google-vertex") while the body reports spaced display names
    # ("Amazon Bedrock", "Google Vertex") -- comparison must fold both.
    assert assert_provider({"provider": "Amazon Bedrock"}, "amazon-bedrock") == "Amazon Bedrock"
    assert assert_provider({"provider": "Google Vertex"}, "google-vertex") == "Google Vertex"


def test_assert_provider_rejects_a_substituted_provider():
    with pytest.raises(ProviderMismatch):
        assert_provider({"provider": "Azure"}, "together")


def test_missing_provider_field_is_an_error_not_a_silent_pass():
    with pytest.raises(ProviderMismatch):
        assert_provider({}, "together")


def test_null_provider_field_is_an_error_not_a_silent_pass():
    with pytest.raises(ProviderMismatch):
        assert_provider({"provider": None}, "together")


def test_assert_provider_check_is_not_a_bare_assert_and_survives_python_dash_O():
    # A bare `assert` is stripped entirely under `python -O`, which would
    # silently disable the only guard against pooling different
    # quantizations of the same model across the comparison. Guard against
    # regressing to `assert` by checking the source has no bare assert
    # statement in this function.
    import inspect
    src = inspect.getsource(assert_provider)
    assert "assert " not in src and not src.strip().startswith("assert")


# ---- resolve_endpoint -----------------------------------------------------

class FakeGetResponse:
    def __init__(self, json_body, status_ok=True):
        self._json = json_body
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._json


class FakeGetSession:
    def __init__(self, response):
        self._response = response
        self.last_call = None

    def get(self, url, headers=None, timeout=None):
        self.last_call = {"url": url, "headers": headers, "timeout": timeout}
        return self._response


# Real endpoint records carry `supported_parameters` (a list of strings),
# `supports_tool_choice` (a dict keyed none/auto/required/function), and a
# `tag` -- the ROUTABLE pin (Ruling 29), distinct from `provider_name` (a
# display name that 404s if sent as a pin). `_tc()` builds a tool-capable
# endpoint; `_no_tc()` an endpoint that omits tool support entirely -- e.g.
# openai/gpt-4o-mini's Azure endpoint, per the live-probe finding behind
# Ruling 26. `tag` defaults to a PLAIN (no "/") slugified `provider_name` so
# every pre-Ruling-29 test's tie-break expectations are unaffected unless a
# test deliberately passes a compound tag to exercise the new behavior.
def _tc(provider_name, quantization, context_length, tag=None):
    return {
        "provider_name": provider_name,
        "tag": tag if tag is not None else provider_name.lower().replace(" ", "-"),
        "quantization": quantization,
        "context_length": context_length,
        "supported_parameters": ["temperature", "top_p", "tools", "max_tokens"],
        "supports_tool_choice": {"none": True, "auto": True, "required": True, "function": True},
    }


def _no_tc(provider_name, quantization, context_length, tag=None):
    return {
        "provider_name": provider_name,
        "tag": tag if tag is not None else provider_name.lower().replace(" ", "-"),
        "quantization": quantization,
        "context_length": context_length,
        "supported_parameters": ["temperature", "top_p", "max_tokens"],  # no "tools"
        "supports_tool_choice": {},
    }


FAKE_ENDPOINTS = [
    _tc("Novita", "bf16", 12288),
    _tc("DeepInfra", "fp8", 131072),
    _tc("AkashML", "fp8", 131072),
]


def test_resolve_endpoint_picks_highest_context_length_deterministically():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": FAKE_ENDPOINTS}}))
    result = resolve_endpoint("meta-llama/llama-3.3-70b-instruct", api_key="sk-test",
                               session=session)
    # DeepInfra and AkashML tie on context_length (131072); both have plain
    # tags, so the alphabetical tie-break still picks AkashML.
    assert result["provider_name"] == "AkashML"
    assert result["tag"] == "akashml"
    assert result["quantization"] == "fp8"
    assert result["context_length"] == 131072
    assert result["supports_tools"] is True
    assert result["candidates"] == FAKE_ENDPOINTS


# ---- Ruling 26: tool-calling capability must be checked before ranking ---

def test_resolve_endpoint_skips_a_higher_context_endpoint_that_lacks_tool_support():
    # This is the exact live-probe failure: openai/gpt-4o-mini's Azure
    # endpoint has the highest context_length but doesn't support tools;
    # ranking by context_length alone would pick it and every real call
    # would 404. The lower-context, tool-capable endpoint must win instead.
    endpoints = [
        _no_tc("Azure", "unknown", 128000),
        _tc("OpenAI", "unknown", 16384),
    ]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    result = resolve_endpoint("openai/gpt-4o-mini", api_key="sk-test", session=session)
    assert result["provider_name"] == "OpenAI"
    assert result["tag"] == "openai"
    assert result["supports_tools"] is True
    # the full, unfiltered candidate list (including the excluded one) is
    # still returned for audit
    assert result["candidates"] == endpoints


def test_resolve_endpoint_raises_named_exception_when_no_endpoint_supports_tools():
    endpoints = [_no_tc("Azure", "unknown", 128000), _no_tc("SomeOther", "unknown", 4096)]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    with pytest.raises(NoToolCapableEndpoint):
        resolve_endpoint("openai/gpt-4o-mini", api_key="sk-test", session=session)


def test_resolve_endpoint_deterministic_tie_break_holds_among_tool_capable_endpoints():
    # A non-tool-capable endpoint with a HIGHER context_length than either
    # tool-capable tied endpoint must not disturb the alphabetical tie-break
    # among the tool-capable survivors.
    endpoints = [
        _no_tc("Zenith", "unknown", 999999),
        _tc("DeepInfra", "fp8", 131072),
        _tc("AkashML", "fp8", 131072),
    ]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    result = resolve_endpoint("m", api_key="sk-test", session=session)
    assert result["provider_name"] == "AkashML"
    assert result["supports_tools"] is True


# ---- Ruling 29: pin by tag, not provider_name ------------------------------
# Live check found the LLM arms 404 on openai/gpt-5.6-luna: the alphabetical
# tie-break picked provider_name == "Amazon Bedrock" (A sorts before O), and
# that display name is not a routable pin -- confirmed against the live API
# ('Amazon Bedrock', 'amazon-bedrock', and 'bedrock' all 404; only the
# endpoint's own `tag` field, e.g. 'amazon-bedrock/us-east-1', or a plain
# tag like 'openai', actually routes).

def test_resolve_endpoint_prefers_a_plain_tag_over_a_compound_one_at_equal_context_length():
    # The exact defect: the alphabetically-first provider ("Amazon Bedrock")
    # has a compound regional tag; a later provider ("OpenAI") has a plain
    # tag. Both tie on context_length. The plain tag must win regardless of
    # alphabetical order.
    endpoints = [
        _tc("Amazon Bedrock", "bf16", 128000, tag="amazon-bedrock/us-east-1"),
        _tc("OpenAI", "bf16", 128000, tag="openai"),
    ]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    result = resolve_endpoint("openai/gpt-5.6-luna", api_key="sk-test", session=session)
    assert result["tag"] == "openai"
    assert result["provider_name"] == "OpenAI"


def test_resolve_endpoint_returns_the_tag_distinct_from_provider_name():
    endpoints = [_tc("Amazon Bedrock", "bf16", 128000, tag="amazon-bedrock/us-east-1")]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    result = resolve_endpoint("m", api_key="sk-test", session=session)
    assert result["tag"] == "amazon-bedrock/us-east-1"
    assert result["provider_name"] == "Amazon Bedrock"
    assert result["tag"] != result["provider_name"]


def test_resolve_endpoint_raises_when_chosen_endpoint_has_no_tag():
    # A raw record missing "tag" entirely -- not just an empty string --
    # must not silently fall back to provider_name as the pin.
    endpoints = [{
        "provider_name": "Amazon Bedrock", "quantization": "bf16", "context_length": 128000,
        "supported_parameters": ["temperature", "top_p", "tools", "max_tokens"],
        "supports_tool_choice": {"auto": True},
    }]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    with pytest.raises(ValueError):
        resolve_endpoint("m", api_key="sk-test", session=session)


def test_the_full_pipeline_pins_the_payload_with_the_tag_not_the_display_name(monkeypatch):
    """End-to-end: resolve_endpoint -> chat() -> build_payload actually sends
    the tag on the wire, never the display name."""
    endpoints = [_tc("Amazon Bedrock", "bf16", 128000, tag="amazon-bedrock/us-east-1")]
    get_session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    resolved = resolve_endpoint("m", api_key="sk-test", session=get_session)
    assert resolved["tag"] == "amazon-bedrock/us-east-1"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()
    resp_body["provider"] = "Amazon Bedrock"  # what the live API actually echoes back
    post_session = FakeSession(FakeResponse(resp_body))

    _, served = chat("m", [], tools=[], provider=resolved["tag"], session=post_session)

    sent_pin = post_session.last_call["json"]["provider"]["order"][0]
    assert sent_pin == "amazon-bedrock/us-east-1"
    assert sent_pin != resolved["provider_name"]  # never the display name
    assert served == "Amazon Bedrock"  # verified correctly despite the tag/display-name split


def test_assert_provider_accepts_a_compound_tag_served_by_its_matching_display_name():
    assert assert_provider({"provider": "OpenAI"}, "openai/flex") == "OpenAI"
    assert assert_provider({"provider": "Amazon Bedrock"},
                           "amazon-bedrock/us-east-1") == "Amazon Bedrock"


def test_assert_provider_rejects_a_compound_tag_served_by_a_different_provider():
    with pytest.raises(ProviderMismatch):
        assert_provider({"provider": "Azure"}, "openai/flex")


def test_resolve_endpoint_prefer_is_scoped_to_tool_capable_endpoints_only():
    # An explicit `prefer` for a provider that CAN'T do tool calling must
    # not be honored -- it would be as unusable as the default pick.
    endpoints = [_no_tc("Azure", "unknown", 128000), _tc("OpenAI", "unknown", 16384)]
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": endpoints}}))
    with pytest.raises(ValueError):
        resolve_endpoint("openai/gpt-4o-mini", api_key="sk-test", session=session, prefer="azure")


def test_resolve_endpoint_is_deterministic_across_repeated_calls():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": FAKE_ENDPOINTS}}))
    first = resolve_endpoint("m", api_key="sk-test", session=session)
    second = resolve_endpoint("m", api_key="sk-test", session=session)
    assert first["provider_name"] == second["provider_name"]


def test_resolve_endpoint_honors_explicit_prefer_override():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": FAKE_ENDPOINTS}}))
    result = resolve_endpoint("m", api_key="sk-test", session=session, prefer="novita")
    assert result["provider_name"] == "Novita"
    assert result["tag"] == "novita"
    assert result["quantization"] == "bf16"
    assert result["context_length"] == 12288


def test_resolve_endpoint_raises_when_prefer_matches_nothing():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": FAKE_ENDPOINTS}}))
    with pytest.raises(ValueError):
        resolve_endpoint("m", api_key="sk-test", session=session, prefer="nonexistent-provider")


def test_resolve_endpoint_raises_on_empty_endpoint_list():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": []}}))
    with pytest.raises(ValueError):
        resolve_endpoint("m", api_key="sk-test", session=session)


def test_resolve_endpoint_sends_bearer_auth_and_no_key_leak_in_url():
    session = FakeGetSession(FakeGetResponse({"data": {"endpoints": FAKE_ENDPOINTS}}))
    resolve_endpoint("meta-llama/llama-3.3-70b-instruct", api_key="sk-secret-sentinel",
                      session=session)
    assert session.last_call["headers"]["Authorization"] == "Bearer sk-secret-sentinel"
    assert "sk-secret-sentinel" not in session.last_call["url"]


# ---- extract_tool_call -----------------------------------------------------

def _response_with_tool_call(arguments_str='{"action": "select", "target": 3}', name="pick"):
    return {
        "choices": [
            {"message": {"tool_calls": [
                {"id": "call_1", "function": {"name": name, "arguments": arguments_str}}
            ]}}
        ]
    }


def test_extract_tool_call_returns_name_arguments_id_on_well_formed_response():
    resp = _response_with_tool_call()
    out = extract_tool_call(resp)
    assert out == {"name": "pick", "arguments": {"action": "select", "target": 3}, "id": "call_1"}


def test_extract_tool_call_raises_on_no_tool_calls():
    resp = {"choices": [{"message": {"tool_calls": []}}]}
    with pytest.raises(ParseFailure):
        extract_tool_call(resp)


def test_extract_tool_call_raises_when_tool_calls_key_absent():
    resp = {"choices": [{"message": {"content": "no tool call here"}}]}
    with pytest.raises(ParseFailure):
        extract_tool_call(resp)


def test_extract_tool_call_raises_on_malformed_response_structure():
    with pytest.raises(ParseFailure):
        extract_tool_call({"choices": []})
    with pytest.raises(ParseFailure):
        extract_tool_call({})
    with pytest.raises(ParseFailure):
        extract_tool_call({"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]})  # no function


def test_extract_tool_call_raises_on_arguments_not_valid_json():
    resp = _response_with_tool_call(arguments_str="{not valid json")
    with pytest.raises(ParseFailure):
        extract_tool_call(resp)


# ---- extract_tool_calls (plural) -----------------------------------------------------

def _response_with_two_tool_calls():
    return {
        "choices": [{"message": {"tool_calls": [
            {"id": "call_1", "function": {"name": "read_buffer", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "execute",
                                           "arguments": '{"action": "send_message"}'}},
        ]}}]
    }


def test_extract_tool_calls_returns_every_call_in_order_not_just_the_first():
    resp = _response_with_two_tool_calls()
    out = extract_tool_calls(resp)
    assert len(out) == 2
    assert out[0] == {"name": "read_buffer", "arguments": {}, "id": "call_1"}
    assert out[1] == {"name": "execute", "arguments": {"action": "send_message"}, "id": "call_2"}


def test_extract_tool_calls_raises_on_no_tool_calls():
    with pytest.raises(ParseFailure):
        extract_tool_calls({"choices": [{"message": {"tool_calls": []}}]})


def test_extract_tool_call_singular_still_returns_only_the_first_of_several():
    resp = _response_with_two_tool_calls()
    assert extract_tool_call(resp) == {"name": "read_buffer", "arguments": {}, "id": "call_1"}


# ---- record_request -----------------------------------------------------

def _response_with_usage(provider="together", generation_id="gen-1"):
    resp = _response_with_tool_call()
    resp["id"] = generation_id
    resp["provider"] = provider
    resp["usage"] = {
        "cost": 0.00123,
        "prompt_tokens": 500,
        "completion_tokens": 42,
        "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 10},
    }
    return resp


def test_record_request_captures_cost_and_cache_token_fields():
    row = record_request("x/y", "together", "together", 0.3, 0.95, 256,
                          _response_with_usage(), quantization="fp8", context_length=131072)
    assert row["cost"] == 0.00123
    assert row["prompt_tokens"] == 500
    assert row["completion_tokens"] == 42
    assert row["cached_tokens"] == 400
    assert row["cache_write_tokens"] == 10
    assert row["quantization"] == "fp8"
    assert row["context_length"] == 131072
    assert row["generation_id"] == "gen-1"


def test_record_request_tolerates_missing_usage_block():
    resp = _response_with_tool_call()
    resp["id"] = "gen-2"
    row = record_request("x/y", "together", "together", 0.3, 0.95, 256, resp)
    assert row["cost"] is None
    assert row["cached_tokens"] is None


def test_record_request_threads_the_tag_alongside_provider_name():
    # Ruling 29: pinned_provider is now the TAG that was actually sent
    # (e.g. "amazon-bedrock/us-east-1"); provider_name is the separate,
    # human-readable display name threaded through for reporting.
    row = record_request("x/y", "amazon-bedrock/us-east-1", "Amazon Bedrock", 0.3, 0.95, 256,
                          _response_with_usage(), provider_name="Amazon Bedrock")
    assert row["pinned_provider"] == "amazon-bedrock/us-east-1"
    assert row["provider_name"] == "Amazon Bedrock"


def test_record_request_provider_name_defaults_to_none():
    row = record_request("x/y", "together", "together", 0.3, 0.95, 256, _response_with_usage())
    assert row["provider_name"] is None


# ---- chat() end-to-end with a fake session -----------------------------------------------------

class FakeResponse:
    def __init__(self, json_body, status_ok=True):
        self._json = json_body
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._json


class FakeSession:
    """Records the last request it was asked to make; returns a canned response."""

    def __init__(self, response):
        self._response = response
        self.last_call = None

    def post(self, url, json=None, headers=None, timeout=None):
        self.last_call = {"url": url, "json": json, "headers": headers, "timeout": timeout}
        return self._response


def test_chat_sends_expected_payload_and_returns_response_and_served_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()
    resp_body["provider"] = "together"
    session = FakeSession(FakeResponse(resp_body))

    response_json, served = chat(
        "x/y", [{"role": "user", "content": "hi"}], tools=[], provider="together",
        temperature=0.3, top_p=0.95, max_tokens=256, session=session,
    )

    assert served == "together"
    assert response_json == resp_body
    sent = session.last_call["json"]
    assert sent["provider"] == {"order": ["together"], "allow_fallbacks": False}
    assert "quantizations" not in sent["provider"]
    assert sent["temperature"] == 0.3
    assert sent["top_p"] == 0.95
    assert sent["max_tokens"] == 256
    assert session.last_call["headers"]["Authorization"] == "Bearer sk-test-sentinel"


def test_chat_verifies_provider_from_body_not_headers(monkeypatch):
    # A FakeResponse with no .headers attribute at all must still work --
    # verification must not touch headers.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()
    resp_body["provider"] = "together"
    assert not hasattr(FakeResponse(resp_body), "headers")
    session = FakeSession(FakeResponse(resp_body))

    _, served = chat("x/y", [], tools=[], provider="together", session=session)
    assert served == "together"


def test_chat_matches_provider_case_insensitively(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()
    resp_body["provider"] = "DeepInfra"
    session = FakeSession(FakeResponse(resp_body))

    _, served = chat("x/y", [], tools=[], provider="deepinfra", session=session)
    assert served == "DeepInfra"


def test_chat_raises_when_served_provider_does_not_match_pin(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()
    resp_body["provider"] = "Azure"
    session = FakeSession(FakeResponse(resp_body))

    with pytest.raises(ProviderMismatch):
        chat("x/y", [], tools=[], provider="together", session=session)


def test_chat_raises_when_provider_field_absent_from_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_tool_call()  # no "provider" key at all
    session = FakeSession(FakeResponse(resp_body))

    with pytest.raises(ProviderMismatch):
        chat("x/y", [], tools=[], provider="together", session=session)


def test_chat_record_hook_receives_a_provenance_row(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_usage(provider="together", generation_id="gen-456")
    session = FakeSession(FakeResponse(resp_body))

    recorded = []
    chat("x/y", [], tools=[], provider="together", temperature=0.1, top_p=0.8,
         max_tokens=128, session=session, record_hook=recorded.append,
         quantization="fp8", context_length=131072)

    assert len(recorded) == 1
    row = recorded[0]
    assert row["model"] == "x/y"
    assert row["pinned_provider"] == "together"
    assert row["served_provider"] == "together"
    assert row["temperature"] == 0.1
    assert row["top_p"] == 0.8
    assert row["max_tokens"] == 128
    assert row["generation_id"] == "gen-456"
    assert row["cost"] == 0.00123
    assert row["cached_tokens"] == 400
    assert row["cache_write_tokens"] == 10
    assert row["quantization"] == "fp8"
    assert row["context_length"] == 131072


def test_chat_threads_provider_name_through_to_the_provenance_record(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-sentinel")
    resp_body = _response_with_usage(provider="Amazon Bedrock", generation_id="gen-789")
    session = FakeSession(FakeResponse(resp_body))

    recorded = []
    chat("m", [], tools=[], provider="amazon-bedrock/us-east-1", session=session,
         record_hook=recorded.append, provider_name="Amazon Bedrock")

    row = recorded[0]
    assert row["pinned_provider"] == "amazon-bedrock/us-east-1"
    assert row["provider_name"] == "Amazon Bedrock"
    assert row["served_provider"] == "Amazon Bedrock"


def test_provenance_row_never_contains_the_api_key(monkeypatch):
    sentinel_key = "sk-super-secret-sentinel-value-do-not-leak"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel_key)
    resp_body = _response_with_usage(provider="together")
    session = FakeSession(FakeResponse(resp_body))

    recorded = []
    chat("x/y", [], tools=[], provider="together", session=session, record_hook=recorded.append)

    row = recorded[0]
    serialized = json.dumps(row)
    assert sentinel_key not in serialized
    for v in row.values():
        assert v != sentinel_key


# ---- opt-in live test -----------------------------------------------------
# Skipped unless a real API key AND an explicit opt-in flag are both present,
# so this never runs in a normal offline suite invocation.

LIVE = bool(os.environ.get("OPENROUTER_API_KEY")) and os.environ.get("NAG_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="live OpenRouter call; set OPENROUTER_API_KEY and NAG_LIVE_TESTS=1 to run")
def test_live_chat_response_body_carries_provider_and_usage_cost():
    response_json, served = chat(
        "meta-llama/llama-3.3-70b-instruct",
        [{"role": "user", "content": "Say 'ok'."}],
        tools=[],
        provider="deepinfra",
        max_tokens=8,
    )
    assert "provider" in response_json
    assert served
    assert "usage" in response_json
    assert "cost" in response_json["usage"]


# --- transient-failure retry (added after the full-arm live smoke lost 5 of
# --- 68 sequential episodes to 429s and one missing `provider` field) -------

class _ScriptedResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._body


class _ScriptedSession:
    """Replays a fixed list of responses and records every request made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self._responses.pop(0)


_OK = {"provider": "Cloudflare", "choices": [{"message": {"content": "ok"}}]}


def _chat(session, **kw):
    slept = []
    out = openrouter.chat(
        model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[], provider="cloudflare", session=session,
        sleep=slept.append, rng=random.Random(0), **kw)
    return out, slept


def test_chat_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([
        _ScriptedResponse(429), _ScriptedResponse(429), _ScriptedResponse(200, _OK)])
    (response_json, served), slept = _chat(session)
    assert served == "Cloudflare"
    assert len(session.calls) == 3        # two retries, not two dropped episodes
    assert len(slept) == 2
    assert all(d >= 0 for d in slept)


def test_chat_honours_retry_after_header(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([
        _ScriptedResponse(429, headers={"Retry-After": "7"}), _ScriptedResponse(200, _OK)])
    _, slept = _chat(session)
    assert slept == [7.0]                 # the server's number, not our backoff


def test_chat_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([_ScriptedResponse(429) for _ in range(4)])
    with pytest.raises(requests.HTTPError):
        _chat(session, max_retries=3)
    assert len(session.calls) == 4        # 1 attempt + 3 retries, then it surfaces


def test_chat_retries_missing_provider_field(monkeypatch):
    """A body with no `provider` is an upstream defect a retry usually clears.

    The live smoke lost `caution:w10` to exactly this. Retrying it is not a
    weakening of the routing guard: an unverifiable response is still never
    accepted, it is merely re-requested first.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([
        _ScriptedResponse(200, {"choices": []}), _ScriptedResponse(200, _OK)])
    (_, served), slept = _chat(session)
    assert served == "Cloudflare"
    assert len(slept) == 1


def test_chat_never_retries_a_provider_substitution(monkeypatch):
    """A substitution means the pin did not hold. Retrying would paper over
    the exact corruption this module exists to detect, so it must surface on
    the first response -- with no second request issued."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([
        _ScriptedResponse(200, {"provider": "DeepInfra", "choices": []}),
        _ScriptedResponse(200, _OK)])
    with pytest.raises(openrouter.ProviderMismatch, match="substituted"):
        _chat(session)
    assert len(session.calls) == 1


def test_chat_records_retry_count_in_provenance(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([_ScriptedResponse(429), _ScriptedResponse(200, _OK)])
    records = []
    openrouter.chat(
        model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[], provider="cloudflare", session=session, record_hook=records.append,
        sleep=lambda d: None, rng=random.Random(0))
    assert len(records) == 1
    assert records[0]["n_retries"] == 1              # reportable, not invisible
    assert "sentinel-key" not in json.dumps(records[0])


def test_chat_records_zero_retries_on_a_clean_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _ScriptedSession([_ScriptedResponse(200, _OK)])
    records = []
    openrouter.chat(
        model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}],
        tools=[], provider="cloudflare", session=session, record_hook=records.append,
        sleep=lambda d: None, rng=random.Random(0))
    assert records[0]["n_retries"] == 0
    assert len(session.calls) == 1


def test_retry_delay_is_jittered_and_capped():
    """Unjittered backoff would resynchronise every retrying request in the
    main run's tight loop onto the same instant, re-tripping the same limit."""
    rng = random.Random(1)
    delays = [openrouter._retry_delay(3, None, rng) for _ in range(20)]
    assert len(set(delays)) > 1
    assert all(0.0 <= d <= openrouter.BACKOFF_CAP_S for d in delays)
    assert openrouter._retry_delay(50, None, rng) <= openrouter.BACKOFF_CAP_S


class _RaisingSession:
    """Raises the given exceptions in order, then serves responses."""

    def __init__(self, raises, responses):
        self._raises = list(raises)
        self._responses = list(responses)
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        if self._raises:
            raise self._raises.pop(0)
        return self._responses.pop(0)


@pytest.mark.parametrize("exc", [
    requests.ConnectionError("read timed out"),
    requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
    requests.ReadTimeout("timed out"),
])
def test_chat_retries_transport_errors(monkeypatch, exc):
    """Transport failures happen BEFORE any status code exists, so a
    status-only retry policy drops every one. The first long run lost 30
    episodes this way across three exception types."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _RaisingSession([exc], [_ScriptedResponse(200, _OK)])
    (_, served), slept = _chat(session)
    assert served == "Cloudflare"
    assert session.calls == 2
    assert len(slept) == 1


def test_chat_surfaces_transport_error_after_max_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-key")
    session = _RaisingSession([requests.ConnectionError("boom")] * 3, [])
    with pytest.raises(requests.ConnectionError):
        _chat(session, max_retries=2)
    assert session.calls == 3
