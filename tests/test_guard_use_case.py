"""Tests del caso de uso post-guard."""


def test_decide_after_guard_when_blocked():
    from features.security.api import decide_after_guard

    assert decide_after_guard({"blocked": True}) == "__end__"


def test_decide_after_guard_when_allowed():
    from features.security.api import decide_after_guard

    assert decide_after_guard({"blocked": False}) == "supervisor"
