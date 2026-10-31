import pytest

from src.core.emotion_schema import EmotionResult
from src.core.graph import (
    _calibrate_analysis,
    has_crisis_signal,
    should_create_alert,
)
from src.core.text_preprocessing import preprocess_text


def _analysis(**overrides) -> EmotionResult:
    data = dict(
        emotion="neutral",
        risk_level="low",
        depression_score=0.1,
        anxiety_score=0.1,
        stress_score=0.1,
        summary="**Support Summary**\n\n**Risk Level:** Low",
        profile_note="",
        user_reply="ok",
        key_characteristics=[],
    )
    data.update(overrides)
    return EmotionResult(**data)


@pytest.mark.parametrize(
    "text",
    [
        "I want to die",
        "i don't want to live anymore",
        "Sometimes I think about suicide",
        "I keep wanting to hurt myself",
        "Я не хочу больше жить",
        "хочется умереть",
        "думаю покончить с собой",
        "Өмір сүргім келмейді",
        "өлгім келеді",
    ],
)
def test_crisis_signals_detected(text):
    assert has_crisis_signal(text)
    assert has_crisis_signal(preprocess_text(text))


@pytest.mark.parametrize(
    "text",
    [
        "I could die laughing at this show",
        "Сегодня обычный день",
        "Один раз я ходил в кино",
        "Однажды мы поехали на море",
    ],
)
def test_no_crisis_signal_in_neutral_text(text):
    assert not has_crisis_signal(text)


def test_crisis_forces_high_risk_and_alert_even_if_model_said_low():
    result = _calibrate_analysis(_analysis(), text="i want to die", language="en", history=[])
    assert result.risk_level == "high"
    assert should_create_alert(result)
    assert "Urgent Support Alert" in result.summary


def test_ambiguous_russian_words_do_not_raise_risk():
    # "один"/"одна" used to match "один раз"/"однажды" and create false alerts.
    text = "однажды я один раз сходил в магазин"
    result = _calibrate_analysis(_analysis(), text=text, language="ru", history=[])
    assert result.risk_level == "low"
    assert not should_create_alert(result)


def test_assistant_history_is_not_a_patient_signal():
    history = ["user: hi", "assistant: You are not alone, I feel the emptiness you describe."]
    result = _calibrate_analysis(_analysis(), text="hello again", language="en", history=history)
    assert result.risk_level == "low"
    assert result.depression_score < 0.7


def test_patient_history_still_counts_for_withdrawal():
    history = ["user: I feel so lonely lately"]
    result = _calibrate_analysis(_analysis(), text="hello", language="en", history=history)
    assert result.risk_level == "medium"
    assert result.depression_score >= 0.7


def test_model_output_is_normalized():
    result = _calibrate_analysis(
        _analysis(risk_level=" High ", emotion="NEGATIVE", depression_score=1.7, anxiety_score=-2),
        text="",
        language="en",
    )
    assert result.risk_level == "high"
    assert result.emotion == "negative"
    assert result.depression_score == 1.0
    assert result.anxiety_score == 0.0


def test_unknown_risk_value_falls_back_to_low():
    result = _calibrate_analysis(_analysis(risk_level="severe??"), text="", language="en")
    assert result.risk_level == "low"


def test_preprocess_keeps_apostrophes_and_hyphens():
    assert preprocess_text("I CAN’T breathe! http://x.y") == "i can't breathe!"
    assert "self-harm" in preprocess_text("Thinking about self-harm 😔")


class _FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    def _respond(self):
        if self.error:
            raise self.error
        return type("Response", (), {"content": self.content})()

    def invoke(self, messages):
        return self._respond()

    async def ainvoke(self, messages):
        return self._respond()


def test_graph_returns_reply_and_analysis(monkeypatch):
    import asyncio
    import json

    from src.core import graph

    payload = {
        "emotion": "negative",
        "risk_level": "Medium",
        "depression_score": 0.4,
        "anxiety_score": 0.3,
        "stress_score": 0.5,
        "summary": "**Support Summary**",
        "profile_note": "tired",
        "user_reply": "That sounds exhausting.",
        "key_characteristics": ["sleep worsens on stressful days"],
    }
    monkeypatch.setattr(graph, "llm_struct", _FakeLLM(content=json.dumps(payload)))
    reply, analysis = asyncio.run(graph.run_graph_with_analysis("work is too much", language="en"))
    assert reply == "That sounds exhausting."
    assert analysis.risk_level == "medium"
    assert analysis.key_characteristics == ["sleep worsens on stressful days"]


def test_graph_llm_failure_keeps_crisis_detection(monkeypatch):
    import asyncio

    from src.core import graph

    monkeypatch.setattr(graph, "llm_struct", _FakeLLM(error=RuntimeError("network down")))
    reply, analysis = asyncio.run(graph.run_graph_with_analysis("Я не хочу жить", language="ru"))
    assert reply  # localized fallback reply
    assert analysis.risk_level == "high"
    assert should_create_alert(analysis)

    dialog = asyncio.run(graph.analyze_dialog(["всё нормально", "хочу умереть"], language="ru"))
    assert dialog.risk_level == "high"
