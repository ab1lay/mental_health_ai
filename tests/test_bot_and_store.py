from datetime import datetime

import pytest

from src.bot import telegram_bot as bot_module
from src.bot.telegram_bot import TelegramBot
from src.storage.dialog_store import DialogStore


@pytest.fixture
def bot():
    # Skip __init__: helpers under test do not need Telegram or the DB.
    return TelegramBot.__new__(TelegramBot)


@pytest.fixture
def store(tmp_path):
    return DialogStore(db_path=tmp_path / "dialogs.db")


class _FixedDatetime(datetime):
    fixed = datetime(2026, 1, 1, 23, 30)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed


@pytest.mark.parametrize(
    "now, start, end, expected",
    [
        ((23, 30), "22:00", "08:00", True),
        ((7, 59), "22:00", "08:00", True),
        ((8, 0), "22:00", "08:00", False),
        ((13, 0), "12:00", "14:00", True),
        ((15, 0), "12:00", "14:00", False),
        ((15, 0), "10:00", "10:00", False),
        ((15, 0), None, "10:00", False),
        ((15, 0), "25:99", "10:00", False),
    ],
)
def test_quiet_hours(bot, monkeypatch, now, start, end, expected):
    _FixedDatetime.fixed = datetime(2026, 1, 1, *now)
    monkeypatch.setattr(bot_module, "datetime", _FixedDatetime)
    assert bot._is_quiet_hours_now(start, end) is expected


def test_smart_interval_shortens_for_rising_risk(bot):
    base = 24
    assert bot._compute_smart_interval_hours(base, []) == 24
    rising = [
        {"risk_level": "high", "depression_score": 0.9},
        {"risk_level": "low", "depression_score": 0.1},
    ]
    falling = list(reversed(rising))
    assert bot._compute_smart_interval_hours(base, rising) < base
    assert bot._compute_smart_interval_hours(base, falling) > base
    assert 6 <= bot._compute_smart_interval_hours(1, rising) <= 168


def test_phone_validation(bot):
    assert bot._is_valid_phone(bot._normalize_phone("+7 (701) 123-45-67"))
    assert bot._is_valid_phone(bot._normalize_phone("0077011234567"))
    assert not bot._is_valid_phone(bot._normalize_phone("12345"))
    assert not bot._is_valid_phone(bot._normalize_phone("+7 abc 123"))


def test_gender_parsing_includes_kazakh(bot):
    assert bot._parse_gender("Женский") == "female"
    assert bot._parse_gender("ер") == "male"
    assert bot._parse_gender("басқа") == "other"
    assert bot._parse_gender("???") is None


@pytest.mark.parametrize("language", ["en", "ru", "kk"])
def test_texts_are_localized_and_not_mojibake(bot, language):
    portal = bot._portal_text(language, "http://x/patient/login?code=ab", "ab", 30)
    assert "??" not in portal
    assert "ab" in portal
    for key in ("full_name", "consent_required", "link_invalid", "link_linked", "error"):
        assert bot._prompt_text(key, language)
    assert bot._crisis_text(language, has_specialist=True)


def test_invite_link_accepts_lowercase_and_reports_status(store):
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO invites (code, user_id, expires_at) VALUES ('AB12CD34', 1, datetime('now', '+1 hour'));"
        )
        conn.execute(
            "INSERT INTO invites (code, user_id, expires_at) VALUES ('DEAD0001', 1, datetime('now', '-1 hour'));"
        )
        conn.commit()

    assert store.link_patient_with_invite(" ab12cd34 ", 42, "Alice", username="alice") == (True, "linked")
    assert store.has_specialist(42)
    assert store.link_patient_with_invite("AB12CD34", 43, "Bob") == (False, "used")
    assert store.link_patient_with_invite("DEAD0001", 43, "Bob") == (False, "expired")
    assert store.link_patient_with_invite("nope", 43, "Bob") == (False, "invalid")
    with store._connect() as conn:
        row = conn.execute("SELECT used_by_username FROM invites WHERE code = 'AB12CD34';").fetchone()
    assert row["used_by_username"] == "alice"


def test_active_alert_tracking(store):
    assert not store.has_active_alert(7)
    store.add_alert(7, reason="high_risk_message", risk_level="high")
    assert store.has_active_alert(7)
    with store._connect() as conn:
        conn.execute("UPDATE alerts SET resolved_at = CURRENT_TIMESTAMP WHERE chat_id = '7';")
        conn.commit()
    assert not store.has_active_alert(7)
