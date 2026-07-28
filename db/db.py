"""SQLite-backed job queue for image generation/editing jobs.

Accessed from two places concurrently: HTTP handlers (srv/server.py, one
thread per request) read statuses, while the background worker (srv/worker.py)
writes results — so WAL mode and a busy_timeout are enabled to avoid
"database is locked" errors.

This service is passive: it never pushes results anywhere on its own, it only
stores and serves them on request. The main chat application polls it (see
utils/image_client.py in the main project).
"""
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from conf import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'queued',      -- queued|processing|done|error
    mode TEXT NOT NULL DEFAULT 'txt2img',        -- txt2img|img2img|inpaint|caption

    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    cfg_scale REAL NOT NULL,
    seed INTEGER,
    strength REAL,

    init_image BLOB,
    mask_image BLOB,
    result_image BLOB,
    result_text TEXT,     -- mode="caption" result (image jobs use result_image instead)
    error_message TEXT,

    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds columns introduced after the initial schema to an existing
    jobs.sqlite3 (CREATE TABLE IF NOT EXISTS alone won't add a column to a
    table that already exists from an earlier version)."""
    if not _column_exists(conn, "jobs", "result_text"):
        conn.execute("ALTER TABLE jobs ADD COLUMN result_text TEXT")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


# ---------- Writing jobs ----------
def create_job(
    prompt: str,
    mode: str = "txt2img",
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.5,
    seed: Optional[int] = None,
    strength: Optional[float] = None,
    init_image: Optional[bytes] = None,
    mask_image: Optional[bytes] = None,
) -> int:
    now = time.time()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (status, mode, prompt, negative_prompt, width, height, steps, "
            "cfg_scale, seed, strength, init_image, mask_image, created_at) "
            "VALUES ('queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mode, prompt, negative_prompt, width, height, steps,
                cfg_scale, seed, strength, init_image, mask_image, now,
            ),
        )
        return cur.lastrowid


# ---------- Reading status/result (for the client) ----------
def get_job_status(job_id: int) -> Optional[Dict[str, Any]]:
    """Includes result_text (populated for finished mode="caption" jobs
    only) so the client can read a text answer straight from the status
    response instead of needing a second round-trip the way image results
    (via GET /jobs/{id}/result) do — text is small enough that bundling it
    here is simpler for both sides."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, status, mode, created_at, started_at, finished_at, "
            "error_message, result_text FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None


def get_job_result(job_id: int) -> Optional[bytes]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT result_image FROM jobs WHERE id = ? AND status = 'done'", (job_id,)
        ).fetchone()
        return row["result_image"] if row else None


def delete_job(job_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


# ---------- Worker-facing operations ----------
def fetch_next_queued() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def mark_processing(job_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'processing', started_at = ? WHERE id = ?",
            (time.time(), job_id),
        )


def mark_done(job_id: int, result_image: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'done', result_image = ?, finished_at = ? WHERE id = ?",
            (result_image, time.time(), job_id),
        )


def mark_done_text(job_id: int, result_text: str) -> None:
    """Same as mark_done() but for mode="caption" jobs, which produce a
    text answer instead of an image."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'done', result_text = ?, finished_at = ? WHERE id = ?",
            (result_text, time.time(), job_id),
        )


def mark_error(job_id: int, error_message: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'error', error_message = ?, finished_at = ? WHERE id = ?",
            (error_message, time.time(), job_id),
        )


def purge_expired(ttl_hours: float) -> int:
    """Deletes finished (done/error) but unclaimed jobs older than ttl_hours.
    Returns the number of deleted rows."""
    cutoff = time.time() - ttl_hours * 3600
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM jobs WHERE status IN ('done', 'error') AND finished_at < ?", (cutoff,)
        )
        return cur.rowcount


def reset_stuck_processing() -> int:
    """On daemon startup (after a crash/restart), jobs stuck in 'processing'
    are reset to 'queued' — they will simply be re-run from scratch."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'processing'"
        )
        return cur.rowcount
