import csv
import json
import os
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional


DB_PATH = os.path.join("data", "results.db")
EXPORT_DIR = "exports"
METRIC_BINARY_THRESHOLD = 40.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def init_db(db_path: str = DB_PATH) -> None:
    _ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hgn_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            subject_name TEXT DEFAULT '',
            stop_time TEXT DEFAULT '',
            arrest_time TEXT DEFAULT '',
            head_warning_count INTEGER DEFAULT 0,
            head_movement_too_much INTEGER DEFAULT 0,
            max_head_movement REAL DEFAULT 0.0,
            lack_of_smooth_pursuit_left_real REAL DEFAULT 0.0,
            lack_of_smooth_pursuit_left_binary INTEGER DEFAULT 0,
            lack_of_smooth_pursuit_right_real REAL DEFAULT 0.0,
            lack_of_smooth_pursuit_right_binary INTEGER DEFAULT 0,
            nystagmus_prior_to_45_left_real REAL DEFAULT 0.0,
            nystagmus_prior_to_45_left_binary INTEGER DEFAULT 0,
            nystagmus_prior_to_45_right_real REAL DEFAULT 0.0,
            nystagmus_prior_to_45_right_binary INTEGER DEFAULT 0,
            distinct_nystagmus_max_deviation_left_real REAL DEFAULT 0.0,
            distinct_nystagmus_max_deviation_left_binary INTEGER DEFAULT 0,
            distinct_nystagmus_max_deviation_right_real REAL DEFAULT 0.0,
            distinct_nystagmus_max_deviation_right_binary INTEGER DEFAULT 0,
            vertical_nystagmus REAL DEFAULT 0.0,
            vertical_nystagmus_binary INTEGER DEFAULT 0,
            payload_json TEXT DEFAULT ''
        )
        """
    )
    conn.commit()
    conn.close()


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    _ensure_dir(os.path.dirname(db_path))
    return sqlite3.connect(db_path)


def _as_row_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[str, Any]:
    if row is None:
        return {}
    return {col[0]: row[i] for i, col in enumerate(cursor.description)} if row else {}


def _session_row_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics", {})
    left = (payload.get("binary") or {}).get("left", {})
    right = (payload.get("binary") or {}).get("right", {})
    left_scores = (payload.get("scores") or {}).get("left", {})
    right_scores = (payload.get("scores") or {}).get("right", {})

    row = {
        "session_id": payload.get("session_id") or _now_utc_iso(),
        "created_at": payload.get("created_at") or _now_utc_iso(),
        "subject_name": payload.get("subject_name", ""),
        "stop_time": payload.get("stop_time", ""),
        "arrest_time": payload.get("arrest_time", ""),
        "head_warning_count": _coerce_int(payload.get("head_warning_count", 0)),
        "head_movement_too_much": _coerce_int(payload.get("head_movement_too_much", 0)),
        "max_head_movement": _coerce_float(payload.get("max_head_movement", 0.0)),
    }

    row["lack_of_smooth_pursuit_left_real"] = _coerce_float(
        left_scores.get("lack_of_smooth_pursuit", 0.0)
    )
    row["lack_of_smooth_pursuit_right_real"] = _coerce_float(
        right_scores.get("lack_of_smooth_pursuit", 0.0)
    )
    row["nystagmus_prior_to_45_left_real"] = _coerce_float(
        left_scores.get("nystagmus_prior_to_45", 0.0)
    )
    row["nystagmus_prior_to_45_right_real"] = _coerce_float(
        right_scores.get("nystagmus_prior_to_45", 0.0)
    )
    row["distinct_nystagmus_max_deviation_left_real"] = _coerce_float(
        left_scores.get("distinct_nystagmus_max_deviation", 0.0)
    )
    row["distinct_nystagmus_max_deviation_right_real"] = _coerce_float(
        right_scores.get("distinct_nystagmus_max_deviation", 0.0)
    )
    row["vertical_nystagmus"] = _coerce_float(metrics.get("vertical_nystagmus", 0.0))

    row["lack_of_smooth_pursuit_left_binary"] = _coerce_int(left.get("lack_of_smooth_pursuit", 0))
    row["lack_of_smooth_pursuit_right_binary"] = _coerce_int(right.get("lack_of_smooth_pursuit", 0))
    row["nystagmus_prior_to_45_left_binary"] = _coerce_int(left.get("nystagmus_prior_to_45", 0))
    row["nystagmus_prior_to_45_right_binary"] = _coerce_int(right.get("nystagmus_prior_to_45", 0))
    row["distinct_nystagmus_max_deviation_left_binary"] = _coerce_int(
        left.get("distinct_nystagmus_max_deviation", 0)
    )
    row["distinct_nystagmus_max_deviation_right_binary"] = _coerce_int(
        right.get("distinct_nystagmus_max_deviation", 0)
    )
    row["vertical_nystagmus_binary"] = (
        1 if _coerce_float(row["vertical_nystagmus"]) >= METRIC_BINARY_THRESHOLD else 0
    )
    row["payload_json"] = json.dumps(payload, ensure_ascii=False)
    return row


def _append_csv(payload_row: Dict[str, Any]) -> None:
    _ensure_dir(EXPORT_DIR)
    export_date = str(date.today())
    csv_path = os.path.join(EXPORT_DIR, f"{export_date}.csv")
    fieldnames = [
        "session_id",
        "created_at",
        "subject_name",
        "stop_time",
        "arrest_time",
        "head_warning_count",
        "head_movement_too_much",
        "max_head_movement",
        "lack_of_smooth_pursuit_left_real",
        "lack_of_smooth_pursuit_left_binary",
        "lack_of_smooth_pursuit_right_real",
        "lack_of_smooth_pursuit_right_binary",
        "nystagmus_prior_to_45_left_real",
        "nystagmus_prior_to_45_left_binary",
        "nystagmus_prior_to_45_right_real",
        "nystagmus_prior_to_45_right_binary",
        "distinct_nystagmus_max_deviation_left_real",
        "distinct_nystagmus_max_deviation_left_binary",
        "distinct_nystagmus_max_deviation_right_real",
        "distinct_nystagmus_max_deviation_right_binary",
        "vertical_nystagmus",
        "vertical_nystagmus_binary",
    ]

    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: payload_row.get(k, 0) for k in fieldnames})


def save_session(payload: Dict[str, Any], db_path: str = DB_PATH) -> str:
    init_db(db_path)
    row = _session_row_from_payload(payload)
    conn = _get_conn(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO hgn_sessions (
            session_id,
            created_at,
            subject_name,
            stop_time,
            arrest_time,
            head_warning_count,
            head_movement_too_much,
            max_head_movement,
            lack_of_smooth_pursuit_left_real,
            lack_of_smooth_pursuit_left_binary,
            lack_of_smooth_pursuit_right_real,
            lack_of_smooth_pursuit_right_binary,
            nystagmus_prior_to_45_left_real,
            nystagmus_prior_to_45_left_binary,
            nystagmus_prior_to_45_right_real,
            nystagmus_prior_to_45_right_binary,
            distinct_nystagmus_max_deviation_left_real,
            distinct_nystagmus_max_deviation_left_binary,
            distinct_nystagmus_max_deviation_right_real,
            distinct_nystagmus_max_deviation_right_binary,
            vertical_nystagmus,
            vertical_nystagmus_binary,
            payload_json
        ) VALUES (
            :session_id,
            :created_at,
            :subject_name,
            :stop_time,
            :arrest_time,
            :head_warning_count,
            :head_movement_too_much,
            :max_head_movement,
            :lack_of_smooth_pursuit_left_real,
            :lack_of_smooth_pursuit_left_binary,
            :lack_of_smooth_pursuit_right_real,
            :lack_of_smooth_pursuit_right_binary,
            :nystagmus_prior_to_45_left_real,
            :nystagmus_prior_to_45_left_binary,
            :nystagmus_prior_to_45_right_real,
            :nystagmus_prior_to_45_right_binary,
            :distinct_nystagmus_max_deviation_left_real,
            :distinct_nystagmus_max_deviation_left_binary,
            :distinct_nystagmus_max_deviation_right_real,
            :distinct_nystagmus_max_deviation_right_binary,
            :vertical_nystagmus,
            :vertical_nystagmus_binary,
            :payload_json
        )
        """,
        row,
    )
    conn.commit()
    conn.close()
    _append_csv(row)
    return row["session_id"]


def list_sessions(limit: int = 50, offset: int = 0, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = _get_conn(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT *
        FROM hgn_sessions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (max(1, int(limit)), max(0, int(offset))),
    )
    rows = [_as_row_dict(cursor, row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_session(session_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = _get_conn(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT *
        FROM hgn_sessions
        WHERE session_id = ?
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _as_row_dict(cursor, row)
