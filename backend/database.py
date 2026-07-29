"""SQLite database setup for EduLens."""

import sqlite3
import json
import hashlib
import secrets
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from backend.security import hash_password

DB_PATH = Path(__file__).parent / "edulens.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,"
            "  password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'teacher',"
            "  created_at TEXT DEFAULT (datetime('now')));"
            "CREATE TABLE IF NOT EXISTS sessions ("
            "  id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_path TEXT,"
            "  uploaded_at TEXT DEFAULT (datetime('now')), duration_sec REAL DEFAULT 0,"
            "  status TEXT DEFAULT 'queued', progress INTEGER DEFAULT 0,"
            "  avg_engagement REAL, avg_collab REAL, class_health REAL,"
            "  collab_verdict TEXT, timeline_json TEXT, students_json TEXT,"
            "  alerts_json TEXT, frames_json TEXT, error_message TEXT,"
            "  user_id TEXT REFERENCES users(id));"
            "CREATE TABLE IF NOT EXISTS alerts ("
            "  id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id),"
            "  student_id TEXT, severity TEXT NOT NULL, message TEXT NOT NULL,"
            "  timestamp REAL NOT NULL, resolved INTEGER DEFAULT 0,"
            "  created_at TEXT DEFAULT (datetime('now')));"
            "CREATE TABLE IF NOT EXISTS audit_logs ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,"
            "  action TEXT NOT NULL, resource TEXT,"
            "  created_at TEXT DEFAULT (datetime('now')));"
            "CREATE TABLE IF NOT EXISTS refresh_tokens ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  token_hash TEXT UNIQUE NOT NULL,"
            "  user_id TEXT NOT NULL REFERENCES users(id),"
            "  expires_at TEXT NOT NULL,"
            "  created_at TEXT DEFAULT (datetime('now')),"
            "  revoked INTEGER DEFAULT 0,"
            "  replaced_by TEXT,"
            "  device_info TEXT,"
            "  ip_address TEXT);"
        )
        # Migrate existing DBs
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN frames_json TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE refresh_tokens ADD COLUMN device_info TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE refresh_tokens ADD COLUMN ip_address TEXT")
        except Exception:
            pass
        demo_users = [
            (
                "usr_teacher",
                "Ms. Priya Sharma",
                "teacher@edulens.ai",
                hash_password("demo123"),
                "teacher",
            ),
            (
                "usr_hod",
                "Dr. Ramesh Naidu",
                "hod@edulens.ai",
                hash_password("demo123"),
                "hod",
            ),
            (
                "usr_principal",
                "Prof. K. Anand",
                "principal@edulens.ai",
                hash_password("demo123"),
                "principal",
            ),
        ]

        conn.executemany(
            """
            INSERT OR IGNORE INTO users
            (id, name, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            demo_users,
        )


def get_session(session_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        # Decode all JSON blobs (frames_json kept separate — can be large)
        for field in ('timeline_json', 'students_json', 'alerts_json'):
            if d.get(field):
                d[field.replace('_json', '')] = json.loads(d[field])
            d.pop(field, None)
        # Remove raw frames_json from main session dict (fetched separately via get_session_frames)
        d.pop('frames_json', None)
        return d


def get_session_frames(session_id):
    """Return parsed frames list for bbox overlay, or [] if not available."""
    with db() as conn:
        row = conn.execute(
            "SELECT frames_json FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not row or not row[0]:
            return []
        try:
            return json.loads(row[0])
        except Exception:
            return []


def list_sessions(limit=20, user_id=None):
    with db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id=? ORDER BY uploaded_at DESC LIMIT ?", (user_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY uploaded_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def update_session(session_id, **kwargs):
    sets = ', '.join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    with db() as conn:
        conn.execute(f"UPDATE sessions SET {sets} WHERE id=?", vals)


def create_session(session_id, filename, user_id):
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, filename, user_id) VALUES (?,?,?)",
            (session_id, filename, user_id)
        )


def get_user_by_email(email):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row is None:
            return None
        return dict(row)
def get_user_by_id(user_id: str):
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=?",
            (user_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

def get_connection_raw():
    return get_connection()


def log_audit(user_id, action, resource=''):
    with db() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, resource) VALUES (?,?,?)",
            (user_id, action, resource)
        )


def check_session_ownership(session_id, user_id):
    """Check if a user owns a session. Returns session dict if owned, None otherwise."""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=? AND user_id=?", (session_id, user_id)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        # Decode JSON blobs
        for field in ('timeline_json', 'students_json', 'alerts_json'):
            if d.get(field):
                d[field.replace('_json', '')] = json.loads(d[field])
            d.pop(field, None)
        d.pop('frames_json', None)
        return d


# Refresh Token Helpers

def hash_refresh_token(token: str) -> str:
    """Hash a refresh token using SHA-256."""
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()


def create_refresh_token(user_id: str, expires_at: str, device_info: str = None, ip_address: str = None) -> str:
    """Generate a new refresh token, store its hash, return the raw token."""
    import secrets
    token = secrets.token_urlsafe(64)
    token_hash = hash_refresh_token(token)
    with db() as conn:
        conn.execute(
            """INSERT INTO refresh_tokens (token_hash, user_id, expires_at, device_info, ip_address)
               VALUES (?, ?, ?, ?, ?)""",
            (token_hash, user_id, expires_at, device_info, ip_address)
        )
    return token


def get_refresh_token(token_hash: str):
    """Look up a refresh token by its hash."""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM refresh_tokens WHERE token_hash=?", (token_hash,)
        ).fetchone()
        return dict(row) if row else None


def revoke_refresh_token(token_hash: str) -> bool:
    """Mark a refresh token as revoked."""
    with db() as conn:
        cur = conn.execute(
            "UPDATE refresh_tokens SET revoked=1 WHERE token_hash=?", (token_hash,)
        )
        return cur.rowcount > 0


def delete_refresh_token(token_hash: str) -> bool:
    """Delete a refresh token from the database."""
    with db() as conn:
        cur = conn.execute(
            "DELETE FROM refresh_tokens WHERE token_hash=?", (token_hash,)
        )
        return cur.rowcount > 0


def replace_refresh_token(old_token_hash: str, new_token_hash: str) -> bool:
    """Mark old refresh token as revoked and replaced by new one (rotation)."""
    with db() as conn:
        cur = conn.execute(
            """UPDATE refresh_tokens 
               SET revoked=1, replaced_by=?
               WHERE token_hash=?""",
            (new_token_hash, old_token_hash)
        )
        return cur.rowcount > 0


def cleanup_expired_tokens():
    """Delete expired and revoked refresh tokens."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ? OR revoked=1",
            (now,)
        )
