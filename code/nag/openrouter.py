"""OpenRouter client with mandatory provider pinning.

Two documented OpenRouter behaviours would silently corrupt a multi-model
comparison, and this module's design exists entirely to defeat them:

  1. Default routing is price-based load balancing across upstream providers,
     and different providers serve different quantizations of the same
     model. Unpinned, a model slug is NOT a fixed object across a run --
     some episodes would hit fp8, others bf16, and we would report them as
     one model. Every request therefore carries
     ``provider: {order: [<pinned>], allow_fallbacks: false}``, and the
     served provider is verified against the pin on every response using the
     response body's top-level ``provider`` field -- NOT a response header.
     (OpenRouter's docs describe an ``openrouter-selected-provider`` header;
     live probing found it does not exist on pinned or unpinned requests.
     The body field is what's actually returned.) A mismatch raises, it does
     not warn.

     ``allow_fallbacks: False`` is confirmed load-bearing against the live
     API: an impossible pin returns HTTP 404 and fails loud, while
     ``allow_fallbacks: True`` on the identical request was silently served
     by a different provider than the one requested. That substitution is
     exactly the corruption this module exists to prevent -- never relax it.

     Quantization is NOT sent as a request filter (an earlier version sent
     ``quantizations: ["bf16"]`` on every request; against the live API this
     404s for providers that only serve other quantizations of the same
     model, e.g. DeepInfra serving fp8 for meta-llama/llama-3.3-70b-instruct
     while Novita serves bf16). Quantization is a property of the provider,
     discovered via ``resolve_endpoint`` and recorded per request, not
     imposed as a filter.
  2. OpenRouter omits absent parameters upstream rather than applying
     defaults, letting each provider apply its own. ``temperature``,
     ``top_p``, and ``max_tokens`` are therefore always sent explicitly, so
     experimental arms never differ for reasons unrelated to the study.

Do not "simplify" either guard away.

Transient upstream failures are retried, not dropped. The full-arm live
smoke (``code/scripts/07_smoke.py``) lost 4 of 68 sequential episodes to
HTTP 429 and 1 more to a response body that omitted the ``provider`` field.
At the main run's scale that is not noise: retries are cheap, and an
unretried 429 silently deletes whichever arm happened to be running when a
burst limit tripped, which is a NON-RANDOM hole in a between-arm contrast.
``chat`` therefore retries 429/5xx and a missing ``provider`` field with
exponential backoff plus jitter, honouring ``Retry-After`` when the server
sends it, and records the retry count in the provenance row so the rate is
reportable rather than invisible. A provider SUBSTITUTION is never retried
-- that is a hard scientific failure, not a transient one.

Parse failures (no parseable tool call in a response) are a pre-specified,
reported quantity -- cells above a 15% failure rate are excluded from the
analysis -- so ``extract_tool_call`` raises ``ParseFailure`` rather than
returning ``None`` or an empty dict; a caller cannot accidentally treat a
parse failure as a no-op.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Callable

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"


class ParseFailure(Exception):
    """Model output could not be parsed into a tool call."""


class ProviderMismatch(Exception):
    """The response was not served by the pinned provider (or was unverifiable).

    Raised explicitly rather than checked with a bare ``assert``: assertions
    are stripped entirely under ``python -O``, and this check is the only
    thing standing between the study and silently pooling different
    quantizations (e.g. fp8 and bf16) of the same model slug together as one
    "model" in the multi-model comparison.
    """


class NoToolCapableEndpoint(Exception):
    """No endpoint for a model supports the tool calling this study requires.

    Every request this module sends carries a ``tools`` array and
    ``tool_choice: "auto"`` (see ``build_payload``). Ranking candidate
    endpoints by ``context_length`` alone, without checking whether an
    endpoint can serve that request at all, can silently pick one that
    can't: confirmed against the live API, ``openai/gpt-4o-mini``'s
    highest-``context_length`` endpoint (Azure) omits ``"tools"`` from
    ``supported_parameters``, while its ``openai`` endpoint carries it. A
    pin to the Azure endpoint then 404s on every real call in the run --
    "No endpoints found for openai/gpt-4o-mini" -- which reads like a
    routing bug, not a capability mismatch. Raised explicitly, naming the
    model, at resolution time rather than letting a model silently drop out
    of the panel mid-run with an opaque error.
    """


def _normalize_provider_name(name: str) -> str:
    """Fold a provider identifier to a comparable form.

    Pins are passed as lowercase slugs (``"deepinfra"``) or, since Ruling
    29, endpoint TAGS whose provider portion is kebab-case
    (``"amazon-bedrock"``, ``"google-vertex"``); the response body reports
    spaced display names (``"DeepInfra"``, ``"Amazon Bedrock"``, ``"Google
    Vertex"``). Comparison must ignore case, spaces, AND hyphens or a
    same-provider match would spuriously "mismatch" against its own pin
    (``"amazon-bedrock"`` vs. ``"Amazon Bedrock"`` differ only in exactly
    those two characters).
    """
    return name.strip().lower().replace(" ", "").replace("-", "")


def _tag_provider_portion(tag: str) -> str:
    """The provider-identifying portion of a pinned endpoint tag.

    Ruling 29: the routable pin is the endpoint's TAG
    (``"openai/flex"``, ``"amazon-bedrock/us-east-1"``), not its display
    name (``provider_name``, e.g. ``"OpenAI"``) -- pinning by display name
    404s against the live API (confirmed: ``'Amazon Bedrock'``,
    ``'amazon-bedrock'``, and ``'bedrock'`` all 404; only the full tag or a
    bare routable slug works). The response body, however, never echoes the
    tag -- only a display name -- so verification (`assert_provider`) must
    strip any ``/region-or-variant`` suffix off the tag before comparing. A
    plain tag with no ``/`` (e.g. ``"openai"``) returns unchanged, so a bare
    provider slug pin (pre-Ruling-29 callers, or any plain-tag provider)
    still compares correctly.
    """
    return tag.split("/", 1)[0]


def build_payload(model, messages, tools, provider, temperature, top_p, max_tokens) -> dict:
    """Build the request body. Provider pin and sampling params are never omitted.

    No ``quantizations`` filter is sent -- see the module docstring for why
    that 404s against the live API for providers serving a different
    quantization of the same model slug.
    """
    payload = {
        "model": model,
        "messages": messages,
        "provider": {"order": [provider], "allow_fallbacks": False},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    # The single-shot arm is defined by having NO tools, and it passes
    # tools=None. Sending "tools": null alongside "tool_choice": "auto" is not
    # the same request as sending neither: providers reject or silently
    # reinterpret it. Omit both keys so a no-tools call is genuinely a no-tools
    # call.
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def assert_provider(response_json: dict, pinned: str) -> str:
    """Verify the response was actually served by the pinned provider.

    Reads the top-level ``provider`` field of the response BODY, not a
    response header -- OpenRouter's documented
    ``openrouter-selected-provider`` header does not exist in practice (live
    probing found no ``openrouter-*`` headers on pinned or unpinned
    requests).

    ``pinned`` is whatever was actually sent as the provider pin -- since
    Ruling 29 that is an endpoint TAG (``"openai/flex"``,
    ``"amazon-bedrock/us-east-1"``), not a bare provider slug, because
    pinning by display name or by a bare provider slug for a
    multi-region/multi-variant provider 404s against the live API. The
    response body never echoes the tag, only a display name
    (``"OpenAI"``, ``"Amazon Bedrock"``), so comparison strips any
    ``/region-or-variant`` suffix off ``pinned`` first
    (`_tag_provider_portion`) and folds both sides through
    `_normalize_provider_name` (case-, space-, and hyphen-insensitive)
    before comparing. A plain tag with no ``/`` compares unchanged, so a
    bare provider slug still works exactly as before.

    A missing field is an error, not a silent pass: without it, routing is
    unverifiable, and an unverifiable pin is no pin at all. Raises
    ``ProviderMismatch``, not a bare ``assert`` -- see that class's
    docstring for why.
    """
    served = response_json.get("provider")
    if served is None:
        raise ProviderUnverifiable("no 'provider' field in response body; routing unverifiable")
    pinned_provider = _tag_provider_portion(pinned)
    if _normalize_provider_name(served) != _normalize_provider_name(pinned_provider):
        raise ProviderMismatch(
            f"provider substituted: pinned {pinned} (provider={pinned_provider}), served {served}"
        )
    return served


def _supports_tool_calling(endpoint: dict) -> bool:
    """Whether an endpoint record can serve this study's tool-calling requests.

    Requires both ``"tools"`` present in ``supported_parameters`` (a list of
    strings) AND ``supports_tool_choice.auto`` true (a dict keyed
    ``none``/``auto``/``required``/``function``) -- every request this
    module sends carries a ``tools`` array with ``tool_choice: "auto"``
    (``build_payload``), so an endpoint missing either cannot serve it.
    """
    supported_params = endpoint.get("supported_parameters") or []
    tool_choice = endpoint.get("supports_tool_choice") or {}
    return "tools" in supported_params and bool(tool_choice.get("auto"))


def resolve_endpoint(model_slug: str, api_key: str, prefer: str | None = None,
                      session: Any = None, timeout: int = 30) -> dict:
    """Discover which provider endpoint(s) serve a model and pick one deterministically.

    Queries ``GET /api/v1/models/{model_slug}/endpoints``, whose response
    body has ``data.endpoints[]``, each with (at least) ``provider_name``,
    ``tag``, ``quantization``, ``context_length``, ``supported_parameters``,
    and ``supports_tool_choice``. Different providers serve different
    quantizations AND different context lengths of the same model slug (e.g.
    for meta-llama/llama-3.3-70b-instruct: Novita 12288 tokens vs. DeepInfra
    131072). An unrecorded short-context endpoint could silently truncate
    prompts mid-run, so the chosen endpoint's context_length is always
    surfaced to the caller to record.

    Candidates are filtered to those that actually support tool calling
    (``_supports_tool_calling``) BEFORE any ranking is applied -- ranking by
    context_length alone can pick an endpoint that 404s on every real
    request this module sends; see ``NoToolCapableEndpoint``'s docstring for
    the live-probe finding that motivated this (Ruling 26). If NO endpoint
    for the model supports tool calling, raises ``NoToolCapableEndpoint``
    naming the model -- never returns an unusable endpoint and never falls
    back silently, so a model that cannot do tool calling fails loudly at
    resolution time, not mid-run.

    Selection among the tool-capable survivors is deterministic so a re-run
    of the same study picks the same endpoint: highest ``context_length``
    first; among ties, a PLAIN tag (no ``/region-or-variant`` suffix, e.g.
    ``"openai"``) is preferred over a compound one (e.g.
    ``"amazon-bedrock/us-east-1"``) -- Ruling 29, fewer moving parts and
    more reproducible across runs; remaining ties broken alphabetically by
    ``provider_name`` (the original, and still the final, determinism
    guarantee). This tie-break is what fixed the live defect that motivated
    Ruling 29: the plain alphabetical tie-break used to pick
    ``provider_name == "Amazon Bedrock"`` over ``"OpenAI"`` purely because
    'A' sorts before 'O', and ``"Amazon Bedrock"`` is not a routable pin at
    all (see ``resolve_endpoint``'s return value and `assert_provider`).
    Pass ``prefer`` (a provider name, matched case-insensitively) to
    override the default selection with a specific provider -- matched only
    among tool-capable endpoints, since an explicit preference for an
    endpoint that can't do this study's work would be as unusable as the
    default pick; raises ``ValueError`` if no tool-capable endpoint matches
    it.

    Returns a dict: ``{"provider_name", "tag", "quantization",
    "context_length", "supports_tools", "candidates"}``. ``tag`` is the
    ROUTABLE pin -- callers must pass ``result["tag"]``, never
    ``result["provider_name"]``, as ``chat()``'s ``provider=`` argument
    (Ruling 29: pinning by display name, e.g. ``"Amazon Bedrock"`` or its
    naive slugification ``"amazon-bedrock"``, confirmed 404 against the live
    API; only the full tag, e.g. ``"amazon-bedrock/us-east-1"``, or a
    provider's plain tag, e.g. ``"openai"``, actually routes).
    ``provider_name`` is kept for reporting and for `assert_provider`'s
    response-body comparison only -- it is never itself sent as a pin.
    Raises ``ValueError`` if the chosen endpoint has no ``tag`` at all
    (pinning would silently fall back to a non-routable identifier).
    ``supports_tools`` is always ``True`` for the endpoint actually chosen
    (it passed the filter above), and ``candidates`` is the full,
    UNFILTERED list of endpoint records as returned by the API, for audit
    purposes -- so a run can show which endpoints existed but were
    excluded, not just which one was picked.
    """
    transport = session if session is not None else requests
    url = f"https://openrouter.ai/api/v1/models/{model_slug}/endpoints"
    r = transport.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    endpoints = body.get("data", {}).get("endpoints", [])
    if not endpoints:
        raise ValueError(f"no endpoints returned for model {model_slug!r}")

    tool_capable = [e for e in endpoints if _supports_tool_calling(e)]
    if not tool_capable:
        raise NoToolCapableEndpoint(
            f"no endpoint for {model_slug!r} supports tool calling "
            f"(checked {len(endpoints)} endpoint(s))"
        )

    if prefer is not None:
        matches = [e for e in tool_capable
                   if _normalize_provider_name(e.get("provider_name", "")) == _normalize_provider_name(prefer)]
        if not matches:
            raise ValueError(
                f"no tool-capable endpoint for {model_slug!r} matches prefer={prefer!r}"
            )
        chosen = matches[0]
    else:
        chosen = sorted(
            tool_capable,
            key=lambda e: (
                -int(e.get("context_length") or 0),
                "/" in (e.get("tag") or ""),  # False (plain tag) sorts before True (compound)
                e.get("provider_name") or "",
            ),
        )[0]

    tag = chosen.get("tag")
    if not tag:
        raise ValueError(
            f"resolved endpoint for {model_slug!r} (provider_name="
            f"{chosen.get('provider_name')!r}) has no 'tag' field -- pinning on "
            "provider_name alone is exactly the non-routable pin Ruling 29 exists to prevent"
        )

    return {
        "provider_name": chosen.get("provider_name"),
        "tag": tag,
        "quantization": chosen.get("quantization"),
        "context_length": chosen.get("context_length"),
        "supports_tools": True,
        "candidates": endpoints,
    }


def _parse_call(call: dict) -> dict:
    """Parse one raw tool-call object into ``{"name", "arguments", "id"}``.

    Arguments are parsed with ``json.loads`` -- never string matching, since
    vendors behind OpenRouter escape JSON differently.
    """
    try:
        function = call["function"]
        name = function["name"]
    except (KeyError, TypeError) as e:
        raise ParseFailure(f"tool call missing name: {e}") from e

    raw_args = function.get("arguments", "") or ""
    try:
        arguments = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        raise ParseFailure(f"tool call arguments not valid JSON: {e}") from e

    return {"name": name, "arguments": arguments, "id": call.get("id")}


def extract_tool_calls(response_json: dict) -> list[dict]:
    """Extract ALL tool calls from a chat-completion response, in order.

    Models routinely emit several tool calls in one message (e.g.
    ``read_buffer`` alongside ``execute``). Returning only the first would
    silently drop the rest with no error -- understating coverage and
    corrupting the primary endpoint. Callers (the agent loop) must iterate
    this, not pick a single call. Raises ``ParseFailure`` (never returns an
    empty list) whenever no tool call can be recovered, so failures stay
    countable against the pre-specified 15% budget instead of being
    swallowed.
    """
    try:
        message = response_json["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise ParseFailure(f"response missing expected message structure: {e}") from e

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise ParseFailure("no tool_calls in response message")

    return [_parse_call(c) for c in tool_calls]


def extract_tool_call(response_json: dict) -> dict:
    """Extract the first tool call from a chat-completion response.

    Returns ``{"name": str, "arguments": dict, "id": str | None}``. Kept for
    callers that only ever expect a single call; prefer
    ``extract_tool_calls`` (plural) wherever a response might legitimately
    carry more than one, which is what the agent loop does.
    """
    return extract_tool_calls(response_json)[0]


def record_request(model, pinned_provider, served_provider, temperature, top_p,
                    max_tokens, response_json: dict, quantization=None,
                    context_length=None, supports_tools=None, provider_name=None) -> dict:
    """Build one provenance row for the run manifest.

    Captures the fields needed to prove every request hit the pinned
    provider with explicit sampling params -- model slug, pinned vs. served
    provider, temperature/top_p/max_tokens, the generation id -- plus the
    resolved endpoint's quantization, context_length, and tool-calling
    support (so a short-context or non-tool-capable endpoint is auditable
    after the fact, not just prevented up front by ``resolve_endpoint``) and
    cost/token accounting pulled from the response body's ``usage`` object:
    ``usage.cost`` is the exact USD cost OpenRouter billed for the call, and
    ``usage.prompt_tokens_details`` carries prompt-caching counts
    (``cached_tokens``, ``cache_write_tokens``) that matter for cost
    analysis across the run. Never includes the API key or any auth header.

    ``pinned_provider`` is whatever was actually sent as the pin -- since
    Ruling 29 that is the resolved endpoint's TAG (``"openai/flex"``), not a
    bare provider slug. ``provider_name`` is the separate, human-readable
    display name from ``resolve_endpoint`` (``"OpenAI"``), threaded through
    alongside ``quantization``/``context_length``/``supports_tools`` purely
    for report readability -- the tag is what actually pinned the request,
    and stays authoritative; ``provider_name`` is never itself sent as a
    pin (see `assert_provider`, `resolve_endpoint`).
    """
    usage = response_json.get("usage") or {}
    cache_details = usage.get("prompt_tokens_details") or {}
    return {
        "model": model,
        "pinned_provider": pinned_provider,
        "provider_name": provider_name,
        "served_provider": served_provider,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "generation_id": response_json.get("id"),
        "quantization": quantization,
        "context_length": context_length,
        "supports_tools": supports_tools,
        "cost": usage.get("cost"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cache_details.get("cached_tokens"),
        "cache_write_tokens": cache_details.get("cache_write_tokens"),
    }



# --- transient-failure retry policy --------------------------------------

class _Transient(Exception):
    """Internal signal that the attempt failed in a retryable way. Never
    escapes `chat` -- the final attempt re-raises the real error instead."""


RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Transport-level failures raised by `requests` BEFORE any status code exists.
# The first long run logged 22 ChunkedEncodingError, 6 ConnectionError and 2
# ReadTimeout across ~5k episodes; none of them are HTTP statuses, so a
# status-only retry policy dropped every one. They are as transient as a 429
# and just as non-random in their effect on a between-arm contrast.
RETRY_EXCEPTIONS = (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError)
MAX_RETRIES = 8
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0


class ProviderUnverifiable(ProviderMismatch):
    """The response body carried no ``provider`` field, so routing could not
    be verified.

    A subclass of `ProviderMismatch` so existing callers that guard against
    unverified routing keep working unchanged, but distinguishable because
    the two mean different things operationally: a SUBSTITUTION is a hard
    scientific failure (the pin did not hold, the model is not a fixed
    object, the episode is unusable), while a MISSING field is an upstream
    response defect that a retry usually clears. `chat` retries this one and
    never retries the other.
    """


def _retry_delay(attempt: int, retry_after: str | None, rng: random.Random) -> float:
    """Seconds to wait before retry `attempt` (0-based).

    Honours a server-sent ``Retry-After`` (seconds form) when present and
    parseable, since the server knows its own limit better than we do.
    Otherwise exponential backoff with full jitter: jitter matters because
    the main run issues requests in a tight loop, and unjittered backoff
    would resynchronise every retrying request onto the same instant and
    re-trip the same burst limit.
    """
    if retry_after:
        try:
            return min(float(retry_after), BACKOFF_CAP_S)
        except (TypeError, ValueError):
            pass
    return rng.uniform(0.0, min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_CAP_S))


def chat(model, messages, tools, provider, temperature=0.7, top_p=1.0,
         max_tokens=1024, timeout=120, session: Any = None,
         record_hook: Callable[[dict], None] | None = None,
         quantization=None, context_length=None, supports_tools=None,
         provider_name=None, max_retries: int = MAX_RETRIES,
         sleep: Callable[[float], None] = time.sleep,
         rng: random.Random | None = None) -> tuple[dict, str]:
    """Send one chat-completion request and verify it landed on the pinned provider.

    `session` defaults to the `requests` module but accepts any object with
    a `.post(url, json=, headers=, timeout=)` method returning an object
    with `.raise_for_status()` and `.json()` -- this is how tests exercise
    the full request path without a live network call. Verification reads
    the response BODY (`assert_provider`), not headers -- see that
    function's docstring.

    `provider` is the actual pin sent on the wire -- since Ruling 29 this
    must be an endpoint TAG (`resolve_endpoint(...)["tag"]`), never
    `resolve_endpoint(...)["provider_name"]` (a display name; pinning by
    display name 404s live). `quantization`, `context_length`,
    `supports_tools`, and `provider_name`, if known from a prior
    `resolve_endpoint` call, are passed through into the provenance record
    for audit purposes only; none of the four affect the request itself.

    Transient failures are retried up to `max_retries` times with jittered
    exponential backoff (`_retry_delay`): HTTP statuses in `RETRY_STATUS`
    (429 and the 5xx family) and a response body missing its ``provider``
    field (`ProviderUnverifiable`). A provider SUBSTITUTION is never
    retried -- it means the pin did not hold, which is a hard failure the
    caller must see. The number of retries actually spent is written into
    the provenance row as ``n_retries`` so the run can report its transient
    failure rate instead of hiding it.

    `sleep` and `rng` are injected so tests can exercise the retry path
    without real delays or nondeterminism.

    The API key is read from `OPENROUTER_API_KEY` at call time and is never
    logged, echoed, or included in the returned response or provenance
    record.
    """
    transport = session if session is not None else requests
    key = os.environ["OPENROUTER_API_KEY"]
    payload = build_payload(model, messages, tools, provider, temperature, top_p, max_tokens)
    headers = {"Authorization": f"Bearer {key}"}
    jitter = rng if rng is not None else random.Random()

    n_retries = 0
    while True:
        retry_after = None
        try:
            try:
                r = transport.post(URL, json=payload, headers=headers, timeout=timeout)
            except RETRY_EXCEPTIONS:
                if n_retries >= max_retries:
                    raise
                raise _Transient("transport error")
            status = getattr(r, "status_code", None)
            if status in RETRY_STATUS and n_retries < max_retries:
                retry_after = (getattr(r, "headers", None) or {}).get("Retry-After")
                raise _Transient(f"HTTP {status}")
            r.raise_for_status()
            response_json = r.json()
            served = assert_provider(response_json, provider)
        except ProviderUnverifiable:
            if n_retries >= max_retries:
                raise
        except _Transient:
            pass
        else:
            break
        sleep(_retry_delay(n_retries, retry_after, jitter))
        n_retries += 1

    if record_hook is not None:
        record = record_request(model, provider, served, temperature, top_p,
                                max_tokens, response_json, quantization=quantization,
                                context_length=context_length, supports_tools=supports_tools,
                                provider_name=provider_name)
        record["n_retries"] = n_retries
        record_hook(record)
    return response_json, served
