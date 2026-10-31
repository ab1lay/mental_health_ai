import os
import re
import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from typing import Dict, List


class DialogStore:
    """Simple SQLite storage for dialog messages and reports."""

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            project_root = Path(__file__).resolve().parents[3]
            db_path = project_root / "data" / "dialogs.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    chat_id TEXT PRIMARY KEY,
                    last_user_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_count INTEGER NOT NULL,
                    emotion TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    depression_score REAL NOT NULL,
                    anxiety_score REAL NOT NULL,
                    stress_score REAL NOT NULL,
                    summary TEXT NOT NULL,
                    profile_note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
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
                CREATE TABLE IF NOT EXISTS user_settings (
                    chat_id TEXT PRIMARY KEY,
                    language TEXT NOT NULL DEFAULT 'en',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                CREATE TABLE IF NOT EXISTS proactive_state (
                    chat_id TEXT PRIMARY KEY,
                    last_outreach_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def add_message(self, chat_id: int | str, role: str, text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, text) VALUES (?, ?, ?);",
                (str(chat_id), role, text),
            )
            conn.commit()

    def get_recent_messages(
        self, chat_id: int | str, limit: int = 5
    ) -> List[Dict[str, str]]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT role, text FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (str(chat_id), limit),
            )
            rows = cursor.fetchall()
        # Return in chronological order (old -> new)
        return [
            {"role": role, "text": text}
            for role, text in reversed(rows)
        ]

    def get_user_messages_after_count(
        self,
        chat_id: int | str,
        after_count: int,
        limit: int,
    ) -> List[str]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT text FROM messages
                WHERE chat_id = ? AND role = 'user'
                ORDER BY id ASC
                LIMIT ? OFFSET ?;
                """,
                (str(chat_id), limit, after_count),
            )
            rows = cursor.fetchall()
        return [text for (text,) in rows]

    def get_user_message_count(self, chat_id: int | str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND role = 'user';",
                (str(chat_id),),
            )
            (count,) = cursor.fetchone()
        return int(count)

    def get_last_report_user_count(self, chat_id: int | str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT last_user_count FROM reports WHERE chat_id = ?;",
                (str(chat_id),),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def set_last_report_user_count(self, chat_id: int | str, count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reports (chat_id, last_user_count)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    last_user_count=excluded.last_user_count,
                    updated_at=CURRENT_TIMESTAMP;
                """,
                (str(chat_id), count),
            )
            conn.commit()

    def add_summary(
        self,
        chat_id: int | str,
        user_count: int,
        summary_text: str,
        emotion: str,
        risk_level: str,
        depression_score: float,
        anxiety_score: float,
        stress_score: float,
        profile_note: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (
                    chat_id,
                    user_count,
                    emotion,
                    risk_level,
                    depression_score,
                    anxiety_score,
                    stress_score,
                    summary,
                    profile_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(chat_id),
                    int(user_count),
                    emotion,
                    risk_level,
                    float(depression_score),
                    float(anxiety_score),
                    float(stress_score),
                    summary_text,
                    profile_note,
                ),
            )
            conn.commit()

    def add_alert(
        self,
        chat_id: int | str,
        reason: str,
        risk_level: str,
        emotion: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alerts (chat_id, reason, risk_level) VALUES (?, ?, ?);",
                (str(chat_id), reason, risk_level),
            )
            conn.commit()
        self.add_event(
            chat_id=str(chat_id),
            event_type="alert_created",
            message=f"Alert created (risk={risk_level}, reason={reason}).",
        )
        if emotion:
            self.add_event(
                chat_id=str(chat_id),
                event_type="summary_added",
                message=f"Summary recorded (risk={risk_level}, emotion={emotion}).",
            )

    def has_active_alert(self, chat_id: int | str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM alerts WHERE chat_id = ? AND resolved_at IS NULL LIMIT 1;",
                (str(chat_id),),
            ).fetchone()
        return row is not None

    def link_patient_with_invite(
        self,
        code: str,
        chat_id: int | str,
        display_name: str,
        username: str | None = None,
    ) -> tuple[bool, str]:
        """Returns (ok, status) where status is one of: invalid, used, expired, linked."""
        # Codes are generated as uppercase hex; patients often type them in lowercase.
        code = (code or "").strip().upper()
        with self._connect() as conn:
            invite = conn.execute(
                """
                SELECT code, user_id, expires_at, used_by_chat_id
                FROM invites
                WHERE code = ?;
                """,
                (code,),
            ).fetchone()
            if not invite:
                return False, "invalid"
            if invite["used_by_chat_id"]:
                return False, "used"
            # SQLite stores timestamps as strings; compare in SQL for simplicity.
            expired = conn.execute(
                "SELECT 1 WHERE datetime('now') > datetime(?);",
                (invite["expires_at"],),
            ).fetchone()
            if expired:
                return False, "expired"

            chat_id_str = str(chat_id)
            patient = conn.execute(
                "SELECT id FROM patients WHERE chat_id = ?;",
                (chat_id_str,),
            ).fetchone()
            if not patient:
                cur = conn.execute(
                    "INSERT INTO patients (chat_id, display_name) VALUES (?, ?);",
                    (chat_id_str, display_name),
                )
                patient_id = int(cur.lastrowid)
            else:
                patient_id = int(patient["id"])

            conn.execute(
                """
                INSERT OR IGNORE INTO assignments (patient_id, user_id)
                VALUES (?, ?);
                """,
                (patient_id, int(invite["user_id"])),
            )
            conn.execute(
                """
                UPDATE invites
                SET used_by_chat_id = ?, used_by_username = ?, used_at = CURRENT_TIMESTAMP
                WHERE code = ?;
                """,
                (chat_id_str, username, code),
            )
            conn.commit()
            self.add_event(
                chat_id=chat_id_str,
                event_type="linked_to_specialist",
                message="Patient linked to specialist via invite.",
            )
            return True, "linked"

    def set_invite_used_by_username(self, code: str, username: str | None) -> None:
        if not username:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE invites
                SET used_by_username = ?
                WHERE code = ?;
                """,
                (username, (code or "").strip().upper()),
            )
            conn.commit()

    def upsert_patient_profile(
        self,
        chat_id: int | str,
        display_name: str,
        full_name: str,
        age: int,
        gender: str,
        phone: str,
        trusted_contact: str,
        notes: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO patients (
                    chat_id,
                    display_name,
                    full_name,
                    age,
                    gender,
                    phone,
                    trusted_contact,
                    notes,
                    consent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    full_name=excluded.full_name,
                    age=excluded.age,
                    gender=excluded.gender,
                    phone=excluded.phone,
                    trusted_contact=excluded.trusted_contact,
                    notes=excluded.notes,
                    consent_at=CURRENT_TIMESTAMP;
                """,
                (
                    str(chat_id),
                    display_name,
                    full_name,
                    int(age),
                    gender,
                    phone,
                    trusted_contact,
                    notes,
                ),
            )
            conn.commit()

    def get_language(self, chat_id: int | str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT language FROM user_settings WHERE chat_id = ?;",
                (str(chat_id),),
            ).fetchone()
            return row["language"] if row else "en"

    def set_language(self, chat_id: int | str, language: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (chat_id, language)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    language=excluded.language,
                    updated_at=CURRENT_TIMESTAMP;
                """,
                (str(chat_id), language),
            )
            conn.commit()

    def get_report_language_for_chat(self, chat_id: int | str) -> str | None:
        with self._connect() as conn:
            # If users or assignments table missing, fallback.
            try:
                row = conn.execute(
                    """
                    SELECT u.report_language
                    FROM assignments a
                    JOIN patients p ON p.id = a.patient_id
                    JOIN users u ON u.id = a.user_id
                    WHERE p.chat_id = ?
                    ORDER BY a.created_at DESC
                    LIMIT 1;
                    """,
                    (str(chat_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
            return row["report_language"] if row and row["report_language"] else None

    def has_specialist(self, chat_id: int | str) -> bool:
        with self._connect() as conn:
            try:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM assignments a
                    JOIN patients p ON p.id = a.patient_id
                    WHERE p.chat_id = ?
                    LIMIT 1;
                    """,
                    (str(chat_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
            return row is not None

    def get_patient_profile(self, chat_id: int | str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT chat_id, display_name, full_name, age, gender, phone,
                       trusted_contact, notes, consent_at
                FROM patients
                WHERE chat_id = ?;
                """,
                (str(chat_id),),
            ).fetchone()
            return dict(row) if row else None

    def add_event(self, chat_id: str, event_type: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (chat_id, event_type, message) VALUES (?, ?, ?);",
                (chat_id, event_type, message),
            )
            conn.commit()

    def upsert_patient_characteristics(
        self,
        chat_id: int | str,
        characteristics: List[str],
    ) -> None:
        chat_id_str = str(chat_id)
        def is_generic(text: str) -> bool:
            lowered = (text or "").strip().lower()
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

        def is_specific(text: str) -> bool:
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

        specific: list[str] = []
        fallback_general: list[str] = []
        generic_rejected: list[str] = []
        rejected_details: list[dict[str, str]] = []
        seen = set()
        for item in characteristics:
            value = (item or "").strip()
            if not value:
                rejected_details.append({"value": "", "reason": "empty"})
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            value = value[:160]
            if is_generic(value):
                generic_rejected.append(value)
                rejected_details.append({"value": value, "reason": "generic"})
            elif is_specific(value):
                specific.append(value)
            else:
                fallback_general.append(value)
                rejected_details.append({"value": value, "reason": "non_specific_fallback"})

        normalized = specific[:8] if specific else fallback_general[:1]
        if not normalized:
            logging.debug(
                "Characteristics DB save skipped: chat_id=%s raw=%s generic_rejected=%s rejected_details=%s",
                chat_id_str,
                characteristics,
                generic_rejected[:3],
                rejected_details[:5],
            )
            return
        with self._connect() as conn:
            for item in normalized:
                conn.execute(
                    """
                    INSERT INTO patient_characteristics (chat_id, characteristic)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id, characteristic) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (chat_id_str, item),
                )
            conn.commit()
        logging.debug(
            "Characteristics DB saved: chat_id=%s raw_count=%s specific=%s fallback=%s rejected=%s saved=%s",
            chat_id_str,
            len(characteristics),
            len(specific),
            len(fallback_general),
            len(generic_rejected),
            normalized,
        )


    def list_recent_characteristics(self, chat_id: int | str, limit: int = 20) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT characteristic
                FROM patient_characteristics
                WHERE chat_id = ?
                ORDER BY datetime(updated_at) DESC, id DESC
                LIMIT ?;
                """,
                (str(chat_id), int(limit)),
            ).fetchall()
        return [str(row["characteristic"]) for row in rows]

    def mark_proactive_outreach(self, chat_id: int | str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proactive_state (chat_id, last_outreach_at)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id) DO UPDATE SET
                    last_outreach_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (str(chat_id),),
            )
            conn.commit()

    def list_inactive_chats(
        self,
        idle_hours: int,
        min_user_messages: int = 2,
        limit: int = 20,
    ) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.chat_id AS chat_id,
                    MAX(datetime(m.created_at)) AS last_message_at,
                    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_message_count,
                    us.language AS language,
                    ps.last_outreach_at AS last_outreach_at
                FROM messages m
                LEFT JOIN user_settings us ON us.chat_id = m.chat_id
                LEFT JOIN proactive_state ps ON ps.chat_id = m.chat_id
                GROUP BY m.chat_id
                HAVING user_message_count >= ?
                   AND datetime(MAX(datetime(m.created_at))) <= datetime('now', ?)
                ORDER BY datetime(MAX(datetime(m.created_at))) ASC
                LIMIT ?;
                """,
                (int(min_user_messages), f"-{int(idle_hours)} hours", int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_summaries(self, chat_id: int | str, limit: int = 5) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT risk_level, depression_score, anxiety_score, stress_score, created_at
                FROM summaries
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (str(chat_id), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_patient_preferences(self, chat_id: int | str) -> Dict:
        defaults = {
            "tone": "supportive",
            "response_length": "medium",
            "support_style": "reflective",
            "proactive_enabled": True,
            "proactive_frequency_hours": 24,
            "quiet_hours_start": None,
            "quiet_hours_end": None,
            "topics_to_avoid": "",
        }
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tone, response_length, support_style,
                       proactive_enabled, proactive_frequency_hours,
                       quiet_hours_start, quiet_hours_end, topics_to_avoid
                FROM patient_preferences
                WHERE chat_id = ?;
                """,
                (str(chat_id),),
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
        self,
        chat_id: int | str,
        *,
        tone: str,
        response_length: str,
        support_style: str,
        proactive_enabled: bool,
        proactive_frequency_hours: int,
        quiet_hours_start: str | None,
        quiet_hours_end: str | None,
        topics_to_avoid: str | None,
        actor_type: str = "system",
        actor_id: str | None = None,
        source: str = "bot",
    ) -> None:
        chat_id_str = str(chat_id)
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
        with self._connect() as conn:
            old_row = conn.execute(
                """
                SELECT tone, response_length, support_style, proactive_enabled,
                       proactive_frequency_hours, quiet_hours_start, quiet_hours_end, topics_to_avoid
                FROM patient_preferences
                WHERE chat_id = ?;
                """,
                (chat_id_str,),
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
                    chat_id_str,
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
                        chat_id_str,
                        actor_type,
                        actor_id,
                        source,
                        json.dumps(changed, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def create_patient_portal_token(self, chat_id: int | str, ttl_minutes: int = 30) -> str:
        with self._connect() as conn:
            code = conn.execute("SELECT lower(hex(randomblob(4))) AS code;").fetchone()["code"]
            conn.execute(
                """
                INSERT INTO patient_portal_tokens (code, chat_id, expires_at)
                VALUES (?, ?, datetime('now', ?));
                """,
                (code, str(chat_id), f"+{int(ttl_minutes)} minutes"),
            )
            conn.commit()
        return str(code)
