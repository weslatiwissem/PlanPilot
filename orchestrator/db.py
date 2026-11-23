"""Users database (SQLite: one file, built into Python, nothing to install).

File: planpilot.db in the project root (change with DATABASE_PATH in .env). Created on first start.

Tables
  users     the whole team, lead included: login + what Agent 2 needs (skills, capacity, load, Trello name)
  sessions  login tokens (one row per logged-in browser, expires after SESSION_DAYS)

Roles
  admin   manages accounts + everything a lead can do. Created automatically from .env on the first start.
  lead    runs the planning steps and approves (writes to Trello)
  member  can log in and see the team

Commands
  python -m orchestrator.db list            # every account (no passwords)
  python -m orchestrator.db create-admin    # add an admin by hand (asks in the terminal)
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from getpass import getpass
from typing import Optional

ROLES = ("admin", "lead", "member")
SESSION_DAYS = 7
PBKDF2_ROUNDS = 200_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'member',
    password_hash   TEXT NOT NULL,
    skills          TEXT NOT NULL DEFAULT '',     -- comma separated
    capacity_points INTEGER NOT NULL DEFAULT 10,  -- story points per sprint
    current_load    INTEGER NOT NULL DEFAULT 0,   -- points already taken elsewhere
    trello_username TEXT,
    in_team         INTEGER NOT NULL DEFAULT 1,   -- 1 = gets stories (an admin may only manage)
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
"""

PUBLIC_FIELDS = ("id", "email", "name", "role", "skills", "capacity_points", "current_load",
                 "trello_username", "in_team")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def connect():
    """`with connect() as conn:` saves the changes and always closes the file (Windows keeps it locked otherwise)."""
    conn = sqlite3.connect(os.getenv("DATABASE_PATH", "planpilot.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    """PBKDF2-SHA256 with a random salt (standard library, no extra package)."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ROUNDS).hex()
    return f"pbkdf2${PBKDF2_ROUNDS}${salt}${digest}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt, digest = stored.split("$")
    except ValueError:
        return False
    test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds)).hex()
    return hmac.compare_digest(test, digest)


# ------------------------------------------------------------------ users
def public(row: sqlite3.Row) -> dict:
    user = {k: row[k] for k in PUBLIC_FIELDS}
    user["skills"] = [s.strip() for s in (row["skills"] or "").split(",") if s.strip()]
    user["in_team"] = bool(row["in_team"])
    return user


def list_users() -> list:
    with connect() as conn:
        return [public(r) for r in conn.execute("SELECT * FROM users ORDER BY role, name")]


def get_user(user_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public(row) if row else None


def create_user(email: str, name: str, password: str, role: str = "member", skills=(), capacity_points: int = 10,
                current_load: int = 0, trello_username: Optional[str] = None, in_team: bool = True) -> dict:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    if len(password) < 6:
        raise ValueError("password must have at least 6 characters")
    with connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, name, role, password_hash, skills, capacity_points, current_load,"
                " trello_username, in_team, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (email.strip().lower(), name.strip(), role, hash_password(password), ", ".join(skills),
                 capacity_points, current_load, trello_username or None, int(in_team), _now().isoformat()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"an account with {email.strip().lower()} already exists")
        new_id = cur.lastrowid
    return get_user(new_id)  # read it back after it is saved


def update_user(user_id: int, **fields) -> Optional[dict]:
    """Change any of: name, role, skills, capacity_points, current_load, trello_username, in_team, password."""
    sets, values = [], []
    for key, value in fields.items():
        if value is None and key != "trello_username":
            continue
        if key == "password":
            if len(value) < 6:
                raise ValueError("password must have at least 6 characters")
            key, value = "password_hash", hash_password(value)
        elif key == "skills":
            value = ", ".join(value)
        elif key == "role" and value not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        elif key == "in_team":
            value = int(value)
        sets.append(f"{key} = ?")
        values.append(value)
    if sets:
        with connect() as conn:
            conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", (*values, user_id))
    return get_user(user_id)


def delete_user(user_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def count_admins() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]


# ------------------------------------------------------------------ login sessions
def login(email: str, password: str) -> Optional[str]:
    """Returns a new token, or None if the email/password is wrong."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if not row or not check_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now().isoformat(),))  # tidy old ones
        conn.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                     (token, row["id"], (_now() + timedelta(days=SESSION_DAYS)).isoformat()))
        return token


def user_for_token(token: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE sessions.token = ? AND sessions.expires_at > ?", (token, _now().isoformat())).fetchone()
    return public(row) if row else None


def logout(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ------------------------------------------------------------------ start-up
def init() -> None:
    """Create the tables, and the admin from .env if there is no account at all yet."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    email, password = os.getenv("ADMIN_EMAIL", "").strip(), os.getenv("ADMIN_PASSWORD", "")
    if empty and email and password:
        create_user(email, os.getenv("ADMIN_NAME", "Admin"), password, role="admin", in_team=False)
        print(f"[planpilot] admin account created for {email}")
    elif empty:
        print("[planpilot] no accounts yet: set ADMIN_EMAIL / ADMIN_PASSWORD in .env and restart, "
              "or run: python -m orchestrator.db create-admin")


def _cli(argv) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    init()
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        users = list_users()
        print(f"{len(users)} account(s) in {os.getenv('DATABASE_PATH', 'planpilot.db')}")
        for u in users:
            print(f"  #{u['id']:<3} {u['role']:<7} {u['name']:<20} {u['email']:<30} "
                  f"skills: {', '.join(u['skills']) or '-'}  capacity: {u['capacity_points']}")
    elif cmd == "create-admin":
        email = input("Admin email: ").strip()
        name = input("Admin name: ").strip() or "Admin"
        password = getpass("Password (min 6 characters): ")
        print("created:", create_user(email, name, password, role="admin", in_team=False)["email"])
    else:
        sys.exit("usage: python -m orchestrator.db [list | create-admin]")


if __name__ == "__main__":
    _cli(sys.argv[1:])
