"""A persistent assistive-action environment for repeated-attempt replay.

Execution here is a STATE CHANGE, not a label. That is the whole point: the
rest of this study scores an action as faithful or unfaithful, which makes a
wrong action look like a classification error. In a deployed assistive system a
wrong action writes something into the world, and `record_refusal` overwriting
`record_consent` is not a misclassification, it is a care preference silently
reversed.

Nothing here communicates externally. There is no network, no file, no clock.

This sandbox is NOT perceived by the agent. Replay drives it from decisions
that were already recorded, so the agent never sees the state it produced and
never learns that a previous attempt failed. Do not describe this as a feedback
loop over the agent; the loop is the retry, not perception.
"""
from __future__ import annotations

import copy

from nag.taxonomy import ACTIONS, TIERS

TIER3_ACTIONS: frozenset[str] = frozenset(a for a in ACTIONS if TIERS[a] == 3)


def _initial_state() -> dict:
    return {
        "notes": [],
        "media_playing": False,
        "light_on": False,
        "outbox": [],
        "call_log": [],
        "posts": [],
        "staff_alert": False,
        "preference_record": None,
    }


class AssistiveSandbox:
    """Nine assistive actions with persistent, inspectable consequences."""

    def __init__(self) -> None:
        self.state = _initial_state()
        self._initial = _initial_state()

    def execute(self, action_name: str) -> None:
        # Refuse rather than no-op: an action present in the frozen taxonomy but
        # missing a handler here would silently score as "executed, no effect",
        # which is the failure this sandbox exists to make visible.
        handler = self._HANDLERS.get(action_name)
        if handler is None:
            raise KeyError(f"no sandbox handler for action {action_name!r}")
        handler(self)

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def changed_from_initial(self) -> bool:
        return self.state != self._initial

    def _save_note(self) -> None:      self.state["notes"].append("note")
    def _play_media(self) -> None:     self.state["media_playing"] = True
    def _set_light(self) -> None:      self.state["light_on"] = True
    def _send_message(self) -> None:   self.state["outbox"].append("message")
    def _place_call(self) -> None:     self.state["call_log"].append("family")
    def _post_update(self) -> None:    self.state["posts"].append("update")
    def _summon_staff(self) -> None:   self.state["staff_alert"] = True
    def _record_refusal(self) -> None: self.state["preference_record"] = "refusal"
    def _record_consent(self) -> None: self.state["preference_record"] = "consent"

    _HANDLERS = {
        "save_note": _save_note, "play_media": _play_media, "set_light": _set_light,
        "send_message": _send_message, "place_call": _place_call,
        "post_update": _post_update, "summon_staff": _summon_staff,
        "record_refusal": _record_refusal, "record_consent": _record_consent,
    }


# A handler for every action in the frozen taxonomy, checked at import.
assert set(AssistiveSandbox._HANDLERS) == set(ACTIONS)
