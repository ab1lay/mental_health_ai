import pytest
from langchain_core.messages import HumanMessage

from src.core import graph


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-6.1-sol", True),
        ("gpt-6-luna", True),
        ("gpt-5-mini", True),
        ("o3-mini", True),
        ("gpt-4o", False),
        ("gpt-4.1-mini", False),
    ],
)
def test_reasoning_model_detection(model, expected):
    assert graph.is_reasoning_model(model) is expected


def _payload(runnable):
    """The exact request body ChatOpenAI would send to the API."""
    messages = [HumanMessage(content="hi")]
    if hasattr(runnable, "bound"):  # RunnableBinding produced by .bind(...)
        return runnable.bound._get_request_payload(messages, **runnable.kwargs)
    return runnable._get_request_payload(messages)


def test_reasoning_model_request_has_no_temperature(monkeypatch):
    monkeypatch.setattr(graph.settings, "model", "gpt-6.1-sol")
    monkeypatch.setattr(graph.settings, "reasoning_effort", "low")
    llm = graph._make_llm(temperature=0.1).bind(response_format={"type": "json_object"})
    payload = _payload(llm)
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "gpt-6.1-sol"


def test_classic_model_keeps_temperature(monkeypatch):
    monkeypatch.setattr(graph.settings, "model", "gpt-4o")
    payload = _payload(graph._make_llm(temperature=0.5))
    assert payload["temperature"] == 0.5
    assert "reasoning_effort" not in payload


def test_module_level_struct_model_uses_json_mode():
    assert _payload(graph.llm_struct)["response_format"] == {"type": "json_object"}
