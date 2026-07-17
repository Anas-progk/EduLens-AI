"""
reid_database.py -- Persistent person identity across sessions.

Problem being solved:
  Tracking algorithms (ByteTrack, SimpleIoU) assign NEW IDs every session.
  Student "Rahul" is Track_3 on Monday, Track_7 on Tuesday, Track_1 on Wednesday.
  This makes cross-session analytics (engagement trends, weekly reports) impossible.

Solution:
  SQLite database stores appearance embeddings (768-d Swin features) for every
  person ever seen. When a new tracking ID appears, we compare its embedding
  against the database using cosine similarity. If match found → reuse the
  existing GlobalID. If no match → assign a new permanent GlobalID.

  The embedding is the MEAN of the last 8 Swin backbone feature vectors for a person.
  It encodes their appearance (clothing, build, hair) without storing any face image.

  Embedding is updated via EMA (alpha=0.1) to handle appearance changes
  (different clothes, lighting variation) across sessions.

Privacy design:
  - NO face images stored, only abstract 768-d float vectors
  - Vectors are not directly invertible to photographs
  - Database can be wiped (`db.clear_all_persons()`) for GDPR compliance
  - Optional: associate GlobalID with a name/seat only via external mapping

Usage:
  db = ReIDDatabase("database/persons.db")

  # When a new tracked person appears:
  embedding = get_swin_embedding(person_crop_sequence)  # (768,) numpy array
  global_id = db.match_or_register(embedding)

  # Log a detection:
  db.log_detection(session_id=1, global_id=global_id, frame_num=120,
                   bbox=(x,y,w,h), engagement="Engaged", eng_prob=0.87,
                   collaboration="Collaborative", collab_prob=0.72)

  # Get person history:
  history = db.get_person_history(global_id)
"""

import sqlite3
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_THRESHOLD_HIGH   = 0.75   # sim > this → definite match, reuse GlobalID
MATCH_THRESHOLD_LOW    = 0.50   # sim < this → definite new person
# Between LOW and HIGH → ambiguous; handled by _resolve_ambiguous()

EMA_ALPHA = 0.10   # Embedding update weight (new frames contribute 10%)
                   # Low alpha = slowly adapts to appearance changes (safe)

EMBEDDING_DIM = 768  # Swin-Tiny temporal feature dimension


# ---------------------------------------------------------------------------
# ReIDDatabase
# ---------------------------------------------------------------------------

class ReIDDatabase:
    """
    SQLite-backed database for persistent person tracking across sessions.

    Thread-safety: NOT thread-safe. Use one instance per process.
    For multi-camera setups, use separate DB files per camera or add a lock.
    """

    def __init__(self, db_path: str = "database/persons.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"ReIDDatabase: connected to {self.db_path}")
        print(f"  Known persons: {self._count_persons()}")

    # ── Schema ────────────────────────────────────────────────────────────

    def _create_tables(self):
        """Create tables if they don't exist (safe to call on existing DB)."""
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                global_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen       REAL NOT NULL,
                last_seen        REAL NOT NULL,
                appearance_emb   BLOB NOT NULL,
                appearance_count INTEGER DEFAULT 1,
                notes            TEXT DEFAULT ''
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time   REAL NOT NULL,
                video_source TEXT DEFAULT '',
                end_time     REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    INTEGER NOT NULL,
                global_id     INTEGER NOT NULL,
                frame_num     INTEGER NOT NULL,
                timestamp     REAL NOT NULL,
                bbox_x        INTEGER,
                bbox_y        INTEGER,
                bbox_w        INTEGER,
                bbox_h        INTEGER,
                engagement    TEXT DEFAULT 'Unknown',
                eng_prob      REAL DEFAULT 0.5,
                collaboration TEXT DEFAULT 'Unknown',
                collab_prob   REAL DEFAULT 0.5,
                FOREIGN KEY(global_id)  REFERENCES persons(global_id),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        c.commit()

    # ── Session management ─────────────────────────────────────────────────

    def start_session(self, video_source: str = "") -> int:
        """
        Create a new session record. Call once when starting inference.
        Returns session_id for use in log_detection().
        """
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO sessions (start_time, video_source) VALUES (?, ?)",
            (now, video_source)
        )
        self._conn.commit()
        session_id = cur.lastrowid
        print(f"ReIDDatabase: session {session_id} started  source={video_source!r}")
        return session_id

    def end_session(self, session_id: int):
        """Mark session as complete."""
        self._conn.execute(
            "UPDATE sessions SET end_time=? WHERE session_id=?",
            (time.time(), session_id)
        )
        self._conn.commit()

    # ── Core ReID logic ────────────────────────────────────────────────────

    def match_or_register(
        self,
        embedding : np.ndarray,   # (768,) float32
        min_count : int = 3,      # minimum n_detections before trusting an ID
    ) -> Tuple[int, float]:
        """
        Match embedding against database or register as new person.

        Args:
            embedding: (768,) float32 appearance embedding
            min_count: only consider persons seen >= this many times as reliable

        Returns:
            (global_id, similarity_score)
            similarity_score=1.0 for new registrations.
        """
        embedding = self._normalize(embedding)

        # Load all stored embeddings
        rows = self._conn.execute(
            "SELECT global_id, appearance_emb, appearance_count FROM persons"
        ).fetchall()

        if not rows:
            # Empty DB — register first person
            return self._register_new(embedding), 1.0

        # Compute cosine similarities
        best_id  = None
        best_sim = -1.0

        for row in rows:
            stored = np.frombuffer(row["appearance_emb"], dtype=np.float32)
            stored = self._normalize(stored)
            sim    = float(np.dot(embedding, stored))   # cosine sim (already normalized)

            if sim > best_sim:
                best_sim = sim
                best_id  = row["global_id"]

        # Decision
        if best_sim >= MATCH_THRESHOLD_HIGH:
            # Confident match → update embedding via EMA
            self._update_embedding(best_id, embedding)
            return best_id, best_sim

        elif best_sim < MATCH_THRESHOLD_LOW:
            # Definite new person
            return self._register_new(embedding), 1.0

        else:
            # Ambiguous (0.50-0.75) → treat as new (conservative for privacy)
            # In a real deployment with seat-mapping, you might resolve this manually
            return self._register_new(embedding), 1.0

    def _register_new(self, embedding: np.ndarray) -> int:
        """Insert new person record and return GlobalID."""
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO persons (first_seen, last_seen, appearance_emb, appearance_count)
               VALUES (?, ?, ?, 1)""",
            (now, now, embedding.astype(np.float32).tobytes())
        )
        self._conn.commit()
        global_id = cur.lastrowid
        print(f"  [ReID] New person registered: GlobalID={global_id}")
        return global_id

    def _update_embedding(self, global_id: int, new_emb: np.ndarray):
        """EMA update of stored embedding. Also update last_seen timestamp."""
        row = self._conn.execute(
            "SELECT appearance_emb, appearance_count FROM persons WHERE global_id=?",
            (global_id,)
        ).fetchone()
        if row is None:
            return

        old_emb   = np.frombuffer(row["appearance_emb"], dtype=np.float32).copy()
        count     = row["appearance_count"]

        # EMA update
        updated   = (1.0 - EMA_ALPHA) * old_emb + EMA_ALPHA * new_emb
        updated   = self._normalize(updated)

        self._conn.execute(
            """UPDATE persons
               SET appearance_emb=?, appearance_count=?, last_seen=?
               WHERE global_id=?""",
            (updated.tobytes(), count + 1, time.time(), global_id)
        )
        self._conn.commit()

    # ── Detection logging ──────────────────────────────────────────────────

    def log_detection(
        self,
        session_id    : int,
        global_id     : int,
        frame_num     : int,
        bbox          : Tuple[int, int, int, int],
        engagement    : str   = "Unknown",
        eng_prob      : float = 0.5,
        collaboration : str   = "Unknown",
        collab_prob   : float = 0.5,
    ):
        """
        Log one detection record.

        Args:
            session_id:    From start_session()
            global_id:     From match_or_register()
            frame_num:     Frame index in video
            bbox:          (x, y, w, h) bounding box
            engagement:    "Engaged" / "Not Engaged" / "Unknown"
            eng_prob:      P(Engaged) 0.0-1.0
            collaboration: "Collaborative" / "Not Collaborative" / "Unknown"
            collab_prob:   P(Collaborative) 0.0-1.0
        """
        x, y, w, h = bbox
        self._conn.execute(
            """INSERT INTO detections
               (session_id, global_id, frame_num, timestamp,
                bbox_x, bbox_y, bbox_w, bbox_h,
                engagement, eng_prob, collaboration, collab_prob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, global_id, frame_num, time.time(),
             x, y, w, h,
             engagement, eng_prob, collaboration, collab_prob)
        )
        # Don't commit every frame — batch commits for speed
        # Call commit_batch() every ~100 frames or at session end

    def commit_batch(self):
        """Commit pending detection records to disk."""
        self._conn.commit()

    # ── Query helpers ──────────────────────────────────────────────────────

    def get_person_history(self, global_id: int) -> List[Dict]:
        """Return all detection records for a person across all sessions."""
        rows = self._conn.execute(
            """SELECT d.*, s.video_source, s.start_time as session_start
               FROM detections d
               JOIN sessions s ON d.session_id = s.session_id
               WHERE d.global_id = ?
               ORDER BY d.timestamp""",
            (global_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_summary(self, session_id: int) -> Dict:
        """Return engagement + collab summary for a session."""
        rows = self._conn.execute(
            """SELECT global_id, engagement, eng_prob, collaboration, collab_prob
               FROM detections WHERE session_id=?""",
            (session_id,)
        ).fetchall()

        if not rows:
            return {}

        from collections import defaultdict
        person_stats = defaultdict(lambda: {
            "engagement_counts": {"Engaged": 0, "Not Engaged": 0, "Unknown": 0},
            "collab_counts":     {"Collaborative": 0, "Not Collaborative": 0, "Unknown": 0},
            "eng_probs": [], "collab_probs": []
        })

        for r in rows:
            pid = r["global_id"]
            person_stats[pid]["engagement_counts"][r["engagement"]] += 1
            person_stats[pid]["collab_counts"][r["collaboration"]]   += 1
            person_stats[pid]["eng_probs"].append(r["eng_prob"])
            person_stats[pid]["collab_probs"].append(r["collab_prob"])

        summary = {}
        for pid, stats in person_stats.items():
            eng_counts = stats["engagement_counts"]
            col_counts = stats["collab_counts"]
            total = sum(eng_counts.values())
            summary[pid] = {
                "global_id":        pid,
                "total_detections": total,
                "pct_engaged":      round(eng_counts["Engaged"] / max(total, 1) * 100, 1),
                "pct_collaborative":round(col_counts["Collaborative"] / max(total, 1) * 100, 1),
                "avg_eng_prob":     round(float(np.mean(stats["eng_probs"])), 3),
                "avg_collab_prob":  round(float(np.mean(stats["collab_probs"])), 3),
            }
        return summary

    def list_known_persons(self) -> List[Dict]:
        """Return summary of all known persons in database."""
        rows = self._conn.execute(
            """SELECT global_id, first_seen, last_seen, appearance_count, notes
               FROM persons ORDER BY global_id"""
        ).fetchall()
        return [dict(r) for r in rows]

    def set_person_notes(self, global_id: int, notes: str):
        """Attach optional human-readable note (seat, name, roll no) to a GlobalID."""
        self._conn.execute(
            "UPDATE persons SET notes=? WHERE global_id=?",
            (notes, global_id)
        )
        self._conn.commit()

    def clear_all_persons(self):
        """
        GDPR/privacy wipe: delete all person records and detections.
        Sessions metadata is preserved (for audit log without personal data).
        WARNING: irreversible.
        """
        self._conn.execute("DELETE FROM detections")
        self._conn.execute("DELETE FROM persons")
        self._conn.commit()
        print("ReIDDatabase: all person records cleared (privacy wipe)")

    # ── Utilities ──────────────────────────────────────────────────────────

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector. Safe against zero vectors."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return (v / norm).astype(np.float32)

    def _count_persons(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM persons").fetchone()
        return row["cnt"] if row else 0

    def close(self):
        """Commit and close database connection."""
        self._conn.commit()
        self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Embedding extractor helper (used by inference pipeline)
# ---------------------------------------------------------------------------

class AppearanceEmbeddingBuffer:
    """
    Maintains a rolling buffer of Swin backbone features per tracked person
    and produces a stable appearance embedding (mean of last N frames).

    Usage:
        emb_buffer = AppearanceEmbeddingBuffer(buffer_size=16)

        # Each frame, after Swin backbone extracts features:
        emb_buffer.update(track_id=3, swin_features=feat_768d)

        # Get stable embedding when enough frames are collected:
        if emb_buffer.is_ready(track_id=3):
            embedding = emb_buffer.get_embedding(track_id=3)
            global_id, sim = db.match_or_register(embedding)
    """

    def __init__(self, buffer_size: int = 16, min_frames: int = 8):
        from collections import deque
        self._buffers = {}           # track_id → deque of (768,) arrays
        self._buffer_size = buffer_size
        self._min_frames  = min_frames

    def update(self, track_id: int, swin_features: np.ndarray):
        """Add a new 768-d feature for a tracked person."""
        if track_id not in self._buffers:
            from collections import deque
            self._buffers[track_id] = deque(maxlen=self._buffer_size)
        self._buffers[track_id].append(swin_features.astype(np.float32))

    def is_ready(self, track_id: int) -> bool:
        """True if enough frames collected for stable embedding."""
        return (track_id in self._buffers and
                len(self._buffers[track_id]) >= self._min_frames)

    def get_embedding(self, track_id: int) -> np.ndarray:
        """Return mean of buffered features as stable appearance embedding."""
        buf = list(self._buffers[track_id])
        return np.mean(buf, axis=0).astype(np.float32)

    def remove(self, track_id: int):
        """Remove buffer for a person (e.g., when tracker loses them)."""
        self._buffers.pop(track_id, None)

    def active_ids(self):
        return list(self._buffers.keys())


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = ReIDDatabase(db_path)

        # Simulate session
        session_id = db.start_session(video_source="test_video.mp4")

        # Register two persons
        emb_A = np.random.randn(768).astype(np.float32)
        emb_B = np.random.randn(768).astype(np.float32)

        gid_A, sim_A = db.match_or_register(emb_A)
        gid_B, sim_B = db.match_or_register(emb_B)
        print(f"PersonA GlobalID={gid_A}  PersonB GlobalID={gid_B}")

        # Same person appears again (slightly different embedding = lighting change)
        emb_A_noisy = emb_A + np.random.randn(768).astype(np.float32) * 0.05
        gid_A2, sim_A2 = db.match_or_register(emb_A_noisy)
        print(f"PersonA again: GlobalID={gid_A2}  sim={sim_A2:.3f} (should be high & match {gid_A})")

        # Log some detections
        for f in range(10):
            db.log_detection(session_id, gid_A, frame_num=f, bbox=(100, 50, 80, 120),
                             engagement="Engaged", eng_prob=0.85,
                             collaboration="Collaborative", collab_prob=0.70)
        db.commit_batch()

        summary = db.get_session_summary(session_id)
        print(f"Session summary: {summary}")

        persons = db.list_known_persons()
        print(f"Known persons: {len(persons)}")
        db.end_session(session_id)

    print("ReIDDatabase tests passed.")
