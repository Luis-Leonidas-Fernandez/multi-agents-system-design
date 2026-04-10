"""Tests del caso de uso post-guard."""


def test_decide_after_guard_when_blocked():
    from application.use_cases.guard_decision import decide_after_guard

    assert decide_after_guard({"blocked": True}) == "__end__"


def test_decide_after_guard_when_allowed():
    from application.use_cases.guard_decision import decide_after_guard

    assert decide_after_guard({"blocked": False}) == "supervisor"
