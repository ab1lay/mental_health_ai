import asyncio
from types import SimpleNamespace

import pytest

from src.bot import telegram_bot as bot_module
from src.bot.telegram_bot import TelegramBot
from src.core.emotion_schema import EmotionResult
from src.storage.dialog_store import DialogStore


class _Message:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append((text, reply_markup))


def _update(text, chat_id=555):
    return SimpleNamespace(
        message=_Message(text),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(full_name="Test User", username="test"),
    )


def _result(risk="low"):
    return EmotionResult(
        emotion="negative" if risk == "high" else "neutral",
        risk_level=risk,
        depression_score=0.9 if risk == "high" else 0.1,
        summary="**Support Summary**",
        user_reply="I'm here with you.",
    )


@pytest.fixture
def bot(tmp_path, monkeypatch):
    instance = TelegramBot.__new__(TelegramBot)
    instance.store = DialogStore(db_path=tmp_path / "dialogs.db")

    async def fake_graph(text, **kwargs):
        risk = "high" if "die" in text else "low"
        return "I'm here with you.", _result(risk)

    async def fake_summary(messages, language="en"):
        return _result("low"), "summary"

    monkeypatch.setattr(bot_module, "run_graph_with_analysis", fake_graph)
    monkeypatch.setattr(bot_module, "summarize_dialog", fake_summary)
    return instance


def _run(bot, update, user_data):
    asyncio.run(bot.handle_message(update, SimpleNamespace(user_data=user_data)))


def test_high_risk_message_triggers_crisis_prompt_and_single_alert(bot):
    user_data = {}
    first = _update("i want to die")
    _run(bot, first, user_data)
    texts = [text for text, _ in first.message.replies]
    assert texts[0] == "I'm here with you."
    assert first.message.replies[1][1] is not None  # crisis keyboard attached
    assert bot.store.has_active_alert(555)

    second = _update("i want to die")
    _run(bot, second, user_data)
    with bot.store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM alerts WHERE chat_id = '555';").fetchone()[0]
    assert count == 1  # no duplicate while the first alert is unresolved
    assert len(second.message.replies) == 2  # but the patient still sees the crisis options


def test_low_risk_messages_produce_summary_without_alert(bot):
    user_data = {}
    for text in ("hi", "a normal day", "thanks"):
        update = _update(text)
        _run(bot, update, user_data)
        assert len(update.message.replies) == 1
    assert bot.store.get_last_report_user_count(555) == 3
    assert not bot.store.has_active_alert(555)


def test_declining_consent_does_not_lock_the_user_out(bot):
    user_data = {"awaiting_consent": True}
    update = _update("нет")
    _run(bot, update, user_data)
    assert not user_data.get("awaiting_consent")
    follow_up = _update("hello")
    _run(bot, follow_up, user_data)
    assert follow_up.message.replies[0][0] == "I'm here with you."


def test_registration_flow_shows_gender_buttons_and_accepts_typed_gender(bot):
    user_data = {"awaiting_age": True}
    age = _update("30")
    _run(bot, age, user_data)
    assert age.message.replies[0][1] is not None  # gender keyboard is sent right away
    gender = _update("female")
    _run(bot, gender, user_data)
    assert user_data["profile_gender"] == "female"
    assert user_data["awaiting_phone"]
