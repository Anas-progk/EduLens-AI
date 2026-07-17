"""SQLite database setup for EduLens."""

import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

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
        )
        # Migrate existing DBs
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN frames_json TEXT")
        except Exception:
            pass
        conn.execute(
            "INSERT OR IGNORE INTO users (id, name, email, password_hash, role) VALUES"
            " ('usr_teacher','Ms. Priya Sharma','teacher@edulens.ai','demo123','teacher'),"
            " ('usr_hod','Dr. Ramesh Naidu','hod@edulens.ai','demo123','hod'),"
            " ('usr_principal','Prof. K. Anand','principal@edulens.ai','demo123','principal')"
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


def list_sessions(limit=20):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY uploaded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_session(session_id, **kwargs):
    sets = ', '.join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    with db() as conn:
        conn.execute(f"UPDATE sessions SET {sets} WHERE id=?", vals)


def create_session(session_id, filename, user_id='usr_teacher'):
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


def get_connection_raw():
    return get_connection()


def log_audit(user_id, action, resource=''):
    with db() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, resource) VALUES (?,?,?)",
            (user_id, action, resource)
        )
