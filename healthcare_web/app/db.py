import os
import re
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR.parent / 'data'
DIALOGS_DB = DATA_DIR / 'dialogs.db'


def get_conn() -> sqlite3.Connection:
    # Ensure data directory exists so SQLite can create/open the DB file.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DIALOGS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return row is not None


def _is_generic_characteristic(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    words = re.findall(r"[a-zA-Z\u0400-\u04FF]+", lowered)
    if not words:
        return True
    generic_words = {
        "positive", "negative", "neutral", "calm", "engaged", "stable", "good",
        "\u043f\u043e\u0437\u0438\u0442\u0438\u0432\u043d\u044b\u0439",
        "\u043d\u0435\u0433\u0430\u0442\u0438\u0432\u043d\u044b\u0439",
        "\u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439",
        "\u0441\u043f\u043e\u043a\u043e\u0439\u043d\u044b\u0439",
        "\u0432\u043e\u0432\u043b\u0435\u0447\u0435\u043d\u043d\u044b\u0439",
        "\u0443\u0432\u043b\u0435\u0447\u0435\u043d\u043d\u044b\u0439",
        "\u0441\u0442\u0430\u0431\u0438\u043b\u044c\u043d\u044b\u0439",
        "\u0445\u043e\u0440\u043e\u0448\u0438\u0439",
    }
    if all(word in generic_words for word in words):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "neutral greeting",
            "initiates conversation",
            "expresses neutral mood",
            "seeks emotional connection",
            "starts conversation",
            "\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043d\u043e\u0441\u0438\u0442 \u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440",
            "\u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0442\u043e\u043d",
            "overall tone",
            "communication is neutral",
        )
    ):
        return True
    return False


def _is_specific_characteristic(text: str) -> bool:
    lowered = (text or "").strip().lower()
    markers = (
        "\u0438\u0437\u0431\u0435\u0433", "\u0442\u0440\u0443\u0434\u043d\u043e", "\u0441\u043b\u043e\u0436\u043d\u043e",
        "\u0442\u0440\u0435\u0432\u043e\u0433", "\u0430\u043d\u0438\u043c\u0435", "\u043c\u0430\u043d\u0433\u0430",
        "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436", "\u0436\u0430\u043d\u0440", "\u043c\u0443\u0437\u044b\u043a",
        "\u0438\u043d\u0442\u0435\u0440\u0435\u0441", "\u043b\u044e\u0431\u0438\u043c", "\u043d\u0440\u0430\u0432",
        "\u0442\u043e\u043b\u043f", "\u0448\u0443\u043c", "\u0442\u0440\u0438\u0433\u0433\u0435\u0440",
        "struggle", "avoids", "worse", "better", "sleep", "appetite", "anxiety", "panic",
        "crowd", "noise", "trigger", "prefers", "interested", "likes", "favorite", "hobby",
        "anime", "manga", "character", "genre", "music", "movie", "book",
    )
    return any(m in lowered for m in markers)


def ensure_tables() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'specialist',
                report_language TEXT NOT NULL DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                full_name TEXT,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                trusted_contact TEXT,
                notes TEXT,
                consent_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, user_id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invites (
                code TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_by_chat_id TEXT,
                used_by_username TEXT,
                used_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_characteristics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                characteristic TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chat_id, characteristic)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_preferences (
                chat_id TEXT PRIMARY KEY,
                tone TEXT NOT NULL DEFAULT 'supportive',
                response_length TEXT NOT NULL DEFAULT 'medium',
                support_style TEXT NOT NULL DEFAULT 'reflective',
                proactive_enabled INTEGER NOT NULL DEFAULT 1,
                proactive_frequency_hours INTEGER NOT NULL DEFAULT 24,
                quiet_hours_start TEXT,
                quiet_hours_end TEXT,
                topics_to_avoid TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_portal_tokens (
                code TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_preference_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                actor_type TEXT NOT NULL,
                actor_id TEXT,
                source TEXT NOT NULL,
                changed_fields TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Backward-compatible migration: add used_by_username if missing.
        try:
            conn.execute("ALTER TABLE invites ADD COLUMN used_by_username TEXT;")
        except sqlite3.OperationalError:
            pass
        # Backward-compatible migration: add report_language if missing.
        try:
            conn.execute("ALTER TABLE users ADD COLUMN report_language TEXT NOT NULL DEFAULT 'en';")
        except sqlite3.OperationalError:
            pass
        # Backward-compatible migration: add patient profile fields if missing.
        for stmt in [
            "ALTER TABLE patients ADD COLUMN full_name TEXT;",
            "ALTER TABLE patients ADD COLUMN age INTEGER;",
            "ALTER TABLE patients ADD COLUMN gender TEXT;",
            "ALTER TABLE patients ADD COLUMN phone TEXT;",
            "ALTER TABLE patients ADD COLUMN trusted_contact TEXT;",
            "ALTER TABLE patients ADD COLUMN notes TEXT;",
            "ALTER TABLE patients ADD COLUMN consent_at TIMESTAMP;",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def sync_patients_from_messages() -> None:
    if not DIALOGS_DB.exists():
        return
    with get_conn() as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages';"
        ).fetchone()
        if not table:
            return
        rows = conn.execute(
            "SELECT DISTINCT chat_id FROM messages ORDER BY chat_id ASC;"
        ).fetchall()
        for row in rows:
            chat_id = row['chat_id']
            display_name = f"Patient {chat_id}"
            conn.execute(
                """
                INSERT OR IGNORE INTO patients (chat_id, display_name)
                VALUES (?, ?);
                """,
                (chat_id, display_name),
            )
        conn.commit()


def get_user_by_email(email: str) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?;",
            (email,),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?;",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def create_user(email: str, password_hash: str, role: str = 'specialist', report_language: str = 'en') -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, role, report_language) VALUES (?, ?, ?, ?);",
            (email, password_hash, role, report_language),
        )
        conn.commit()
        return int(cur.lastrowid)


def set_user_report_language(user_id: int, language: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET report_language = ? WHERE id = ?;",
            (language, user_id),
        )
        conn.commit()


def get_user_report_language(user_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT report_language FROM users WHERE id = ?;",
            (user_id,),
        ).fetchone()
        return row["report_language"] if row and row["report_language"] else "en"


def list_patients_for_user(user: Dict) -> List[Dict]:
    with get_conn() as conn:
        if user['role'] == 'admin':
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY id DESC;"
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT p.* FROM patients p
                JOIN assignments a ON a.patient_id = p.id
                WHERE a.user_id = ?
                ORDER BY p.id DESC;
                """,
                (user['id'],),
            ).fetchall()
        return [dict(r) for r in rows]


def user_can_access_patient(user: Dict, patient: Dict) -> bool:
    if user.get('role') == 'admin':
        return True
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM assignments WHERE patient_id = ? AND user_id = ?;",
            (patient['id'], user['id']),
        ).fetchone()
        return row is not None


def get_patient(patient_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?;",
            (patient_id,),
        ).fetchone()
        return dict(row) if row else None


def get_patient_by_chat_id(chat_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "patients"):
            return None
        row = conn.execute(
            "SELECT * FROM patients WHERE chat_id = ?;",
            (chat_id,),
        ).fetchone()
        return dict(row) if row else None


def get_patient_language(chat_id: str) -> str:
    with get_conn() as conn:
        if not _table_exists(conn, "user_settings"):
            return "en"
        row = conn.execute(
            "SELECT language FROM user_settings WHERE chat_id = ?;",
            (chat_id,),
        ).fetchone()
        lang = str(row["language"]).lower() if row and row["language"] else "en"
        return lang if lang in {"en", "ru", "kk"} else "en"


def list_summaries(chat_id: str, limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "summaries"):
            return []
        rows = conn.execute(
            """
            SELECT * FROM summaries
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def list_messages(chat_id: str, limit: int = 200) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "messages"):
            return []
        rows = conn.execute(
            """
            SELECT role, text, created_at FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, limit),
        ).fetchall()
        items = [dict(r) for r in rows]
        items.reverse()
        return items


def list_patient_characteristics(chat_id: str, limit: int = 20) -> List[str]:
    def select_best(items: List[str]) -> List[str]:
        specific = [i for i in items if not _is_generic_characteristic(i) and _is_specific_characteristic(i)]
        if specific:
            logging.debug(
                "Characteristics web select: chat_id=%s specific=%s fallback=0 total=%s",
                chat_id,
                len(specific),
                len(items),
            )
            return specific
        fallback = [i for i in items if not _is_generic_characteristic(i)]
        logging.debug(
            "Characteristics web select: chat_id=%s specific=0 fallback=%s total=%s selected=%s",
            chat_id,
            len(fallback),
            len(items),
            fallback[:1],
        )
        return fallback[:1]

    with get_conn() as conn:
        if _table_exists(conn, "patient_characteristics"):
            rows = conn.execute(
                """
                SELECT characteristic
                FROM patient_characteristics
                WHERE chat_id = ?
                ORDER BY datetime(updated_at) DESC, id DESC
                LIMIT ?;
                """,
                (chat_id, int(limit)),
            ).fetchall()
            all_items = [str(r["characteristic"]) for r in rows if r["characteristic"]]
            return select_best(all_items)
        if not _table_exists(conn, "summaries"):
            return []
        rows = conn.execute(
            """
            SELECT profile_note
            FROM summaries
            WHERE chat_id = ? AND profile_note IS NOT NULL AND TRIM(profile_note) <> ''
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, int(limit)),
        ).fetchall()
        all_items = [str(r["profile_note"]) for r in rows if r["profile_note"]]
        return select_best(all_items)


def get_patient_preferences(chat_id: str) -> Dict:
    defaults = {
        "tone": "supportive",
        "response_length": "medium",
        "support_style": "reflective",
        "proactive_enabled": True,
        "proactive_frequency_hours": 24,
        "quiet_hours_start": "",
        "quiet_hours_end": "",
        "topics_to_avoid": "",
    }
    with get_conn() as conn:
        if not _table_exists(conn, "patient_preferences"):
            return defaults
        row = conn.execute(
            """
            SELECT tone, response_length, support_style,
                   proactive_enabled, proactive_frequency_hours,
                   quiet_hours_start, quiet_hours_end, topics_to_avoid
            FROM patient_preferences
            WHERE chat_id = ?;
            """,
            (chat_id,),
        ).fetchone()
        if not row:
            return defaults
        data = dict(row)
        data["proactive_enabled"] = bool(data.get("proactive_enabled", 1))
        data["proactive_frequency_hours"] = int(data.get("proactive_frequency_hours") or 24)
        for key, value in defaults.items():
            if data.get(key) is None:
                data[key] = value
        return data


def upsert_patient_preferences(
    chat_id: str,
    *,
    tone: str,
    response_length: str,
    support_style: str,
    proactive_enabled: bool,
    proactive_frequency_hours: int,
    quiet_hours_start: str | None,
    quiet_hours_end: str | None,
    topics_to_avoid: str | None,
    actor_type: str = "patient",
    actor_id: str | None = None,
    source: str = "patient_web",
) -> None:
    new_state = {
        "tone": tone,
        "response_length": response_length,
        "support_style": support_style,
        "proactive_enabled": bool(proactive_enabled),
        "proactive_frequency_hours": int(proactive_frequency_hours),
        "quiet_hours_start": quiet_hours_start or "",
        "quiet_hours_end": quiet_hours_end or "",
        "topics_to_avoid": (topics_to_avoid or "").strip(),
    }
    with get_conn() as conn:
        old_row = conn.execute(
            """
            SELECT tone, response_length, support_style, proactive_enabled,
                   proactive_frequency_hours, quiet_hours_start, quiet_hours_end, topics_to_avoid
            FROM patient_preferences
            WHERE chat_id = ?;
            """,
            (chat_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO patient_preferences (
                chat_id, tone, response_length, support_style,
                proactive_enabled, proactive_frequency_hours,
                quiet_hours_start, quiet_hours_end, topics_to_avoid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                tone=excluded.tone,
                response_length=excluded.response_length,
                support_style=excluded.support_style,
                proactive_enabled=excluded.proactive_enabled,
                proactive_frequency_hours=excluded.proactive_frequency_hours,
                quiet_hours_start=excluded.quiet_hours_start,
                quiet_hours_end=excluded.quiet_hours_end,
                topics_to_avoid=excluded.topics_to_avoid,
                updated_at=CURRENT_TIMESTAMP;
            """,
            (
                chat_id,
                tone,
                response_length,
                support_style,
                1 if proactive_enabled else 0,
                int(proactive_frequency_hours),
                quiet_hours_start,
                quiet_hours_end,
                topics_to_avoid or "",
            ),
        )
        old_state = {
            "tone": old_row["tone"] if old_row else None,
            "response_length": old_row["response_length"] if old_row else None,
            "support_style": old_row["support_style"] if old_row else None,
            "proactive_enabled": bool(old_row["proactive_enabled"]) if old_row else None,
            "proactive_frequency_hours": int(old_row["proactive_frequency_hours"]) if old_row else None,
            "quiet_hours_start": (old_row["quiet_hours_start"] or "") if old_row else None,
            "quiet_hours_end": (old_row["quiet_hours_end"] or "") if old_row else None,
            "topics_to_avoid": (old_row["topics_to_avoid"] or "").strip() if old_row else None,
        }
        changed = {}
        for key, new_value in new_state.items():
            if old_state.get(key) != new_value:
                changed[key] = {"old": old_state.get(key), "new": new_value}
        if changed:
            conn.execute(
                """
                INSERT INTO patient_preference_history (
                    chat_id, actor_type, actor_id, source, changed_fields
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (
                    chat_id,
                    actor_type,
                    actor_id,
                    source,
                    json.dumps(changed, ensure_ascii=False),
                ),
            )
        conn.commit()


def list_patient_preference_history(chat_id: str, limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "patient_preference_history"):
            return []
        rows = conn.execute(
            """
            SELECT actor_type, actor_id, source, changed_fields, created_at
            FROM patient_preference_history
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, int(limit)),
        ).fetchall()
    items: List[Dict] = []
    for row in rows:
        item = dict(row)
        try:
            item["changed_fields"] = json.loads(item.get("changed_fields") or "{}")
        except Exception:
            item["changed_fields"] = {}
        items.append(item)
    return items


def consume_patient_portal_token(code: str) -> str | None:
    if not code:
        return None
    normalized = code.strip().lower()
    with get_conn() as conn:
        if not _table_exists(conn, "patient_portal_tokens"):
            return None
        # Single conditional UPDATE so two concurrent logins cannot both use the code.
        cur = conn.execute(
            """
            UPDATE patient_portal_tokens
            SET used_at = CURRENT_TIMESTAMP
            WHERE code = ? AND used_at IS NULL AND datetime('now') <= datetime(expires_at);
            """,
            (normalized,),
        )
        conn.commit()
        if cur.rowcount != 1:
            return None
        row = conn.execute(
            "SELECT chat_id FROM patient_portal_tokens WHERE code = ?;",
            (normalized,),
        ).fetchone()
        return str(row["chat_id"]) if row else None


def _chat_filter(chat_ids: Optional[List[str]]) -> tuple[str, list]:
    """SQL fragment restricting rows to the given chat ids (None = no restriction)."""
    if chat_ids is None:
        return "", []
    if not chat_ids:
        return " AND 0", []
    placeholders = ", ".join("?" for _ in chat_ids)
    return f" AND chat_id IN ({placeholders})", [str(c) for c in chat_ids]


def list_alerts(
    active_only: bool = True,
    limit: int = 100,
    chat_ids: Optional[List[str]] = None,
) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "alerts"):
            return []
        where = "WHERE 1=1"
        if active_only:
            where += " AND resolved_at IS NULL"
        chat_sql, params = _chat_filter(chat_ids)
        rows = conn.execute(
            f"SELECT * FROM alerts {where}{chat_sql} ORDER BY id DESC LIMIT ?;",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_alert_by_id(alert_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "alerts"):
            return None
        row = conn.execute(
            "SELECT * FROM alerts WHERE id = ?;",
            (alert_id,),
        ).fetchone()
        return dict(row) if row else None


def resolve_alert(alert_id: int) -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "alerts"):
            return
        conn.execute(
            "UPDATE alerts SET resolved_at = CURRENT_TIMESTAMP WHERE id = ?;",
            (alert_id,),
        )
        conn.commit()


def add_audit_event(chat_id: str, event_type: str, message: str) -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "audit_log"):
            return
        conn.execute(
            "INSERT INTO audit_log (chat_id, event_type, message) VALUES (?, ?, ?);",
            (chat_id, event_type, message),
        )
        conn.commit()


def get_last_summary(chat_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "summaries"):
            return None
        row = conn.execute(
            """
            SELECT * FROM summaries
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT 1;
            """,
            (chat_id,),
        ).fetchone()
        return dict(row) if row else None


def count_patients() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM patients;").fetchone()
        return int(row['c']) if row else 0


def count_active_alerts(chat_ids: Optional[List[str]] = None) -> int:
    with get_conn() as conn:
        if not _table_exists(conn, "alerts"):
            return 0
        chat_sql, params = _chat_filter(chat_ids)
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM alerts WHERE resolved_at IS NULL{chat_sql};",
            params,
        ).fetchone()
        return int(row['c']) if row else 0


def clear_summaries() -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "summaries"):
            return
        conn.execute("DELETE FROM summaries;")
        conn.commit()


def clear_messages() -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "messages"):
            return
        conn.execute("DELETE FROM messages;")
        conn.commit()


def clear_summaries_for_chat(chat_id: str) -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "summaries"):
            return
        conn.execute("DELETE FROM summaries WHERE chat_id = ?;", (chat_id,))
        conn.commit()


def clear_messages_for_chat(chat_id: str) -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "messages"):
            return
        conn.execute("DELETE FROM messages WHERE chat_id = ?;", (chat_id,))
        conn.commit()


def clear_timeline_for_chat(chat_id: str) -> None:
    with get_conn() as conn:
        if _table_exists(conn, "audit_log"):
            conn.execute("DELETE FROM audit_log WHERE chat_id = ?;", (chat_id,))
        if _table_exists(conn, "alerts"):
            conn.execute("DELETE FROM alerts WHERE chat_id = ?;", (chat_id,))
        if _table_exists(conn, "summaries"):
            conn.execute("DELETE FROM summaries WHERE chat_id = ?;", (chat_id,))
        conn.commit()


def list_audit_events(chat_id: str, limit: int = 100) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "audit_log"):
            return []
        rows = conn.execute(
            """
            SELECT event_type, message, created_at
            FROM audit_log
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def list_timeline(chat_id: str, limit: int = 100) -> List[Dict]:
    with get_conn() as conn:
        queries = []
        params: list[str] = []
        if _table_exists(conn, "summaries"):
            queries.append(
                """
                SELECT 'summary' AS event_type,
                       summary AS message,
                       created_at AS event_time
                FROM summaries
                WHERE chat_id = ?
                """
            )
            params.append(chat_id)
        if _table_exists(conn, "alerts"):
            queries.append(
                """
                SELECT 'alert_created' AS event_type,
                       'Alert: ' || risk_level || ' (' || reason || ')' AS message,
                       created_at AS event_time
                FROM alerts
                WHERE chat_id = ?
                """
            )
            params.append(chat_id)
            queries.append(
                """
                SELECT 'alert_resolved' AS event_type,
                       'Alert resolved: ' || risk_level || ' (' || reason || ')' AS message,
                       resolved_at AS event_time
                FROM alerts
                WHERE chat_id = ? AND resolved_at IS NOT NULL
                """
            )
            params.append(chat_id)
        if _table_exists(conn, "audit_log"):
            queries.append(
                """
                SELECT event_type AS event_type,
                       message AS message,
                       created_at AS event_time
                FROM audit_log
                WHERE chat_id = ?
                """
            )
            params.append(chat_id)
        if not queries:
            return []
        sql = (
            "SELECT * FROM (" + " UNION ALL ".join(queries) + ") "
            "ORDER BY datetime(event_time) DESC "
            "LIMIT ?;"
        )
        params.append(str(limit))
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def create_invite(user_id: int, ttl_hours: int = 48) -> str:
    with get_conn() as conn:
        code = conn.execute(
            "SELECT hex(randomblob(4)) AS code;"
        ).fetchone()["code"]
        conn.execute(
            """
            INSERT INTO invites (code, user_id, expires_at)
            VALUES (?, ?, datetime('now', ?));
            """,
            (code, user_id, f"+{ttl_hours} hours"),
        )
        conn.commit()
        return code


def deactivate_invite(code: str, user_id: int) -> None:
    with get_conn() as conn:
        if not _table_exists(conn, "invites"):
            return
        conn.execute(
            """
            UPDATE invites
            SET expires_at = datetime('now', '-1 minute')
            WHERE code = ? AND user_id = ?;
            """,
            (code, user_id),
        )
        conn.commit()


def list_invites(user_id: int, limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        if not _table_exists(conn, "invites"):
            return []
        try:
            rows = conn.execute(
                """
                SELECT code, expires_at, used_by_chat_id, used_by_username, used_at, created_at
                FROM invites
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (user_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                """
                SELECT code, expires_at, used_by_chat_id, used_at, created_at
                FROM invites
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (user_id, limit),
            ).fetchall()
        items = [dict(r) for r in rows]
        for item in items:
            expired = conn.execute(
                "SELECT 1 WHERE datetime('now') > datetime(?);",
                (item["expires_at"],),
            ).fetchone()
            item["is_expired"] = bool(expired)
            item["is_used"] = bool(item["used_at"])
        return items
