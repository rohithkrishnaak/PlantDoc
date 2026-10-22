"""User accounts, sessions, profiles and diagnosis history (SQLite, stdlib only).

Passwords: scrypt with a per-user salt. Sessions: random 256-bit tokens, only their
SHA-256 is stored, so a leaked database cannot be replayed as a cookie.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "data/plantdoc.db"))
SESSION_TTL = 30 * 24 * 3600
MAX_HISTORY = 100
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")

PROFILE_DEFAULTS = {
    "phone": "", "state": "", "district": "", "language": "en", "farm_acres": 1.0,
    "crops": ["tomato"], "irrigation": "", "soil": "", "organic_only": False, "theme": "auto",
}
IRRIGATION = {"", "drip", "flood", "sprinkler", "rainfed"}
SOILS = {"", "red", "black", "alluvial", "loam", "sandy", "clay"}
THEMES = {"auto", "light", "dark"}
CROPS = {"tomato", "potato"}


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


@contextmanager
def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL DEFAULT '',
          pw TEXT NOT NULL, profile TEXT NOT NULL DEFAULT '{}', created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(
          token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS history(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          created REAL NOT NULL, status TEXT NOT NULL, disease TEXT, confidence REAL, trust REAL,
          quality REAL, ctx TEXT NOT NULL DEFAULT '{}', thumb TEXT);
        CREATE INDEX IF NOT EXISTS ix_hist_user ON history(user_id, created DESC);
        """)
        c.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))


# ---- passwords -------------------------------------------------------------
def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def _verify(password: str, stored: str) -> bool:
    try:
        salt_b64, _ = stored.split("$")
        return hmac.compare_digest(_hash(password, base64.b64decode(salt_b64)), stored)
    except Exception:
        return False


def _check_password(pw: str) -> None:
    if len(pw) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if len(pw) > 128:
        raise AuthError("Password is too long.")


# ---- users -----------------------------------------------------------------
def _public(row: sqlite3.Row) -> dict:
    prof = {**PROFILE_DEFAULTS, **json.loads(row["profile"] or "{}")}
    return {"id": row["id"], "email": row["email"], "name": row["name"], "profile": prof,
            "created": row["created"]}


def register(email: str, password: str, name: str = "") -> int:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    _check_password(password)
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AuthError("An account with this email already exists.", 409)
        cur = c.execute("INSERT INTO users(email,name,pw,created) VALUES(?,?,?,?)",
                        (email, name.strip()[:80], _hash(password), time.time()))
        return cur.lastrowid


def login(email: str, password: str) -> int:
    with db() as c:
        row = c.execute("SELECT id, pw FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    # Always run a hash so response time does not reveal whether the email exists.
    ok = _verify(password, row["pw"] if row else _hash("x"))
    if not row or not ok:
        raise AuthError("Incorrect email or password.", 401)
    return row["id"]


def get_user(user_id: int) -> dict | None:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _public(row) if row else None


def update_profile(user_id: int, data: dict) -> dict:
    user = get_user(user_id)
    prof = user["profile"]
    name = str(data.get("name", user["name"])).strip()[:80]

    def s(key, limit=60):
        if key in data:
            prof[key] = str(data[key]).strip()[:limit]

    for k in ("phone", "state", "district"):
        s(k)
    if "phone" in data and prof["phone"] and not re.fullmatch(r"[+\d][\d\s-]{6,17}", prof["phone"]):
        raise AuthError("Phone number looks invalid.")
    if "language" in data:
        prof["language"] = data["language"] if data["language"] in ("en", "hi", "ta") else "en"
    if "farm_acres" in data:
        try:
            prof["farm_acres"] = max(0.1, min(float(data["farm_acres"]), 5000.0))
        except (TypeError, ValueError):
            raise AuthError("Farm size must be a number of acres.")
    if "crops" in data:
        crops = [c for c in data["crops"] if c in CROPS] if isinstance(data["crops"], list) else []
        prof["crops"] = crops or ["tomato"]
    if "irrigation" in data:
        prof["irrigation"] = data["irrigation"] if data["irrigation"] in IRRIGATION else ""
    if "soil" in data:
        prof["soil"] = data["soil"] if data["soil"] in SOILS else ""
    if "organic_only" in data:
        prof["organic_only"] = bool(data["organic_only"])
    if "theme" in data:
        prof["theme"] = data["theme"] if data["theme"] in THEMES else "auto"
    with db() as c:
        c.execute("UPDATE users SET name=?, profile=? WHERE id=?", (name, json.dumps(prof), user_id))
    return get_user(user_id)


def change_password(user_id: int, current: str, new: str, keep_token_hash: str | None = None) -> None:
    _check_password(new)
    with db() as c:
        row = c.execute("SELECT pw FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not _verify(current, row["pw"]):
            raise AuthError("Current password is incorrect.", 403)
        c.execute("UPDATE users SET pw=? WHERE id=?", (_hash(new), user_id))
        # Sign out every other device.
        c.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (user_id, keep_token_hash or ""))


def delete_account(user_id: int, password: str) -> None:
    with db() as c:
        row = c.execute("SELECT pw FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not _verify(password, row["pw"]):
            raise AuthError("Password is incorrect.", 403)
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


# ---- sessions --------------------------------------------------------------
def _th(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db() as c:
        c.execute("INSERT INTO sessions VALUES(?,?,?)", (_th(token), user_id, time.time() + SESSION_TTL))
    return token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    with db() as c:
        row = c.execute("SELECT user_id FROM sessions WHERE token_hash=? AND expires>?",
                        (_th(token), time.time())).fetchone()
    return get_user(row["user_id"]) if row else None


def end_session(token: str | None) -> None:
    if token:
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (_th(token),))


# ---- history ---------------------------------------------------------------
def add_history(user_id: int, *, status: str, disease: str | None, confidence: float | None,
                trust: float | None, quality: float | None, ctx: dict, thumb: str | None) -> int:
    with db() as c:
        cur = c.execute(
            "INSERT INTO history(user_id,created,status,disease,confidence,trust,quality,ctx,thumb)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (user_id, time.time(), status, disease, confidence, trust, quality, json.dumps(ctx), thumb))
        c.execute("DELETE FROM history WHERE user_id=? AND id NOT IN "
                  "(SELECT id FROM history WHERE user_id=? ORDER BY created DESC LIMIT ?)",
                  (user_id, user_id, MAX_HISTORY))
        return cur.lastrowid


def _hrow(r: sqlite3.Row, full: bool = False) -> dict:
    d = {"id": r["id"], "created": r["created"], "status": r["status"], "disease": r["disease"],
         "confidence": r["confidence"], "trust": r["trust"], "quality": r["quality"], "thumb": r["thumb"]}
    if full:
        d["ctx"] = json.loads(r["ctx"] or "{}")
    return d


def list_history(user_id: int) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM history WHERE user_id=? ORDER BY created DESC", (user_id,)).fetchall()
    return [_hrow(r) for r in rows]


def get_history(user_id: int, hid: int) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM history WHERE id=? AND user_id=?", (hid, user_id)).fetchone()
    return _hrow(r, full=True) if r else None


def delete_history(user_id: int, hid: int | None = None) -> None:
    with db() as c:
        if hid is None:
            c.execute("DELETE FROM history WHERE user_id=?", (user_id,))
        else:
            c.execute("DELETE FROM history WHERE id=? AND user_id=?", (hid, user_id))
