"""Automated tests for the web back end (no model needed).

Tests:
1. Expert system: urgency, treatment track, dose scaling, escalation, safe fallback.
2. Expert system: every disease x language renders complete text (EN, HI, TA).
3. Accounts: registration rules, login, sessions, profile validation, password change.
4. Accounts: diagnosis history isolation, cap and cascade delete.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import accounts
from src.accounts import AuthError
from src.expert_system import META, build_plan
from src.i18n import LANGUAGES


def _fresh_db() -> None:
    accounts.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
    accounts.init_db()


def _raises(fn, status: int | None = None) -> bool:
    try:
        fn()
    except AuthError as e:
        return status is None or e.status == status
    return False


# 1. Expert system
def test_late_blight_in_wet_weather_is_urgent_and_escalated():
    plan = build_plan("Tomato Late Blight", "en", brown_ratio=0.12,
                      ctx={"humidity": "high", "rain": True})
    assert plan["urgency"]["level"] == 4
    assert plan["track"]["id"] == "integrated"
    assert plan["escalate"] and plan["escalate"]["phone"] == "1800-180-1551"
    assert plan["safety"], "chemical plans must carry safety guidance"


def test_healthy_leaf_gets_preventive_plan_without_chemicals():
    plan = build_plan("Tomato Healthy", "en")
    assert plan["kind"] == "healthy" and plan["track"]["id"] == "preventive"
    assert plan["urgency"]["level"] == 0
    assert all(t["kind"] == "organic" for t in plan["treatment"]) and not plan["safety"]


def test_organic_only_excludes_chemicals():
    plan = build_plan("Tomato Early Blight", "en", ctx={"organic_only": True, "severity": "mild"})
    assert plan["track"]["id"] == "organic"
    assert plan["treatment"] and all(t["kind"] == "organic" for t in plan["treatment"])


def test_severity_override_beats_photo_estimate():
    plan = build_plan("Tomato Early Blight", "en", brown_ratio=0.01, ctx={"severity": "severe"})
    sev = next(f for f in plan["facts"] if f["key"] == "severity")
    assert sev["level"] == "severe" and sev["source"] == "you"
    photo = build_plan("Tomato Early Blight", "en", brown_ratio=0.01)
    assert next(f for f in photo["facts"] if f["key"] == "severity")["source"] == "photo"


def test_doses_scale_with_farm_size():
    one = build_plan("Tomato Early Blight", "en", ctx={"acres": 1})
    five = build_plan("Tomato Early Blight", "en", ctx={"acres": 5})
    assert five["spray_volume"]["total"] == 5 * one["spray_volume"]["total"]
    assert one["treatment"][0]["total"] != five["treatment"][0]["total"]


def test_uncertain_diagnosis_holds_spraying_and_requires_expert():
    plan = build_plan(None, "en", uncertain=True, candidates=["Tomato Early Blight"])
    assert plan["kind"] == "uncertain" and plan["track"]["id"] == "hold"
    assert plan["treatment"] == [] and plan["escalate"]["level"] == "required"
    assert plan["actions"][0]["id"] == "hold"


def test_inputs_are_sanitised():
    plan = build_plan("Potato Late Blight", "xx", ctx={"acres": 10**9, "humidity": "??", "stage": "??"})
    assert plan["ctx"]["acres"] == 5000.0 and plan["ctx"]["humidity"] == "medium"
    assert plan["ctx"]["stage"] == "vegetative"


def test_every_disease_renders_in_every_language():
    for disease in META:
        for lang in LANGUAGES:
            plan = build_plan(disease, lang, ctx={"humidity": "high", "rain": True, "stage": "harvest", "acres": 6})
            texts = [plan["headline"], plan["urgency"]["label"], plan["track"]["label"], plan["disclaimer"]]
            texts += [a["title"] for a in plan["actions"]] + [a["detail"] for a in plan["actions"]]
            texts += [s["task"] for s in plan["schedule"]] + plan["prevention"] + plan["safety"]
            texts += [r["why"] for r in plan["reasoning"]] + [t["name"] for t in plan["treatment"]]
            assert all(isinstance(x, str) and x.strip() for x in texts), (disease, lang)
    hi, en = build_plan("Tomato Late Blight", "hi"), build_plan("Tomato Late Blight", "en")
    assert hi["headline"] != en["headline"]


# 2. Accounts
def test_registration_rules():
    _fresh_db()
    assert _raises(lambda: accounts.register("not-an-email", "longenough1"), 400)
    assert _raises(lambda: accounts.register("a@b.in", "short"), 400)
    uid = accounts.register("A@B.in", "longenough1", "Asha")
    assert accounts.get_user(uid)["email"] == "a@b.in"
    assert _raises(lambda: accounts.register("a@b.in", "longenough1"), 409)


def test_login_and_sessions():
    _fresh_db()
    uid = accounts.register("a@b.in", "longenough1")
    assert accounts.login("a@b.in", "longenough1") == uid
    assert _raises(lambda: accounts.login("a@b.in", "wrongpass1"), 401)
    assert _raises(lambda: accounts.login("nobody@b.in", "longenough1"), 401)
    token = accounts.create_session(uid)
    assert accounts.user_for_token(token)["id"] == uid
    accounts.end_session(token)
    assert accounts.user_for_token(token) is None and accounts.user_for_token(None) is None


def test_profile_is_validated():
    _fresh_db()
    uid = accounts.register("a@b.in", "longenough1")
    prof = accounts.update_profile(uid, {"farm_acres": 12, "crops": ["potato", "wheat"], "theme": "neon",
                                          "language": "fr", "irrigation": "laser"})["profile"]
    assert prof["farm_acres"] == 12 and prof["crops"] == ["potato"]
    assert prof["theme"] == "auto" and prof["language"] == "en" and prof["irrigation"] == ""
    assert _raises(lambda: accounts.update_profile(uid, {"phone": "abc"}), 400)
    assert _raises(lambda: accounts.update_profile(uid, {"farm_acres": "lots"}), 400)


def test_password_change_signs_out_other_devices():
    _fresh_db()
    uid = accounts.register("a@b.in", "longenough1")
    here, other = accounts.create_session(uid), accounts.create_session(uid)
    assert _raises(lambda: accounts.change_password(uid, "wrong", "newpassword1"), 403)
    accounts.change_password(uid, "longenough1", "newpassword1", keep_token_hash=accounts._th(here))
    assert accounts.user_for_token(here) and accounts.user_for_token(other) is None
    assert accounts.login("a@b.in", "newpassword1") == uid and _raises(lambda: accounts.login("a@b.in", "longenough1"))


# 3. History
def _add(uid: int, disease: str = "Tomato Early Blight") -> int:
    return accounts.add_history(uid, status="TRUSTED", disease=disease, confidence=0.9, trust=0.8, quality=0.7,
                                ctx={"ctx": {}, "preds": [[disease, 0.9]]}, thumb=None)


def test_history_is_private_capped_and_cascades():
    _fresh_db()
    a, b = accounts.register("a@b.in", "longenough1"), accounts.register("b@b.in", "longenough1")
    hid = _add(a)
    assert accounts.get_history(b, hid) is None and accounts.list_history(b) == []
    accounts.delete_history(b, hid)                      # someone else's id: no effect
    assert accounts.get_history(a, hid) is not None
    for _ in range(accounts.MAX_HISTORY + 5):
        _add(a)
    assert len(accounts.list_history(a)) == accounts.MAX_HISTORY
    accounts.delete_account(a, "longenough1")
    assert accounts.get_user(a) is None and accounts.list_history(a) == []
    assert accounts.list_history(b) == []


if __name__ == "__main__":
    print("Running web back-end tests...")
    test_late_blight_in_wet_weather_is_urgent_and_escalated()
    print("[PASS] Late blight in wet weather: act today + expert escalation")
    test_healthy_leaf_gets_preventive_plan_without_chemicals()
    print("[PASS] Healthy leaf: preventive plan, no chemicals")
    test_organic_only_excludes_chemicals()
    print("[PASS] Organic-only plans exclude chemicals")
    test_severity_override_beats_photo_estimate()
    print("[PASS] Farmer-set severity overrides the photo estimate")
    test_doses_scale_with_farm_size()
    print("[PASS] Spray volume and doses scale with acres")
    test_uncertain_diagnosis_holds_spraying_and_requires_expert()
    print("[PASS] Uncertain diagnosis: hold spraying, expert required")
    test_inputs_are_sanitised()
    print("[PASS] Out-of-range inputs are clamped")
    test_every_disease_renders_in_every_language()
    print("[PASS] All 6 diseases render complete plans in EN / HI / TA")
    test_registration_rules()
    print("[PASS] Registration validates email, password, duplicates")
    test_login_and_sessions()
    print("[PASS] Login, wrong password and session revoke")
    test_profile_is_validated()
    print("[PASS] Profile fields are validated")
    test_password_change_signs_out_other_devices()
    print("[PASS] Password change signs out other devices")
    test_history_is_private_capped_and_cascades()
    print("[PASS] History is private, capped and deleted with the account")
    print("\n[SUCCESS] ALL WEB BACK-END TESTS PASSED SUCCESSFULLY!")
