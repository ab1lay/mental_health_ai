from __future__ import annotations
import json
from typing import Dict, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.core.text_preprocessing import preprocess_text
from src.core.emotion_schema import EmotionResult
from config.settings import settings


class GraphState(TypedDict, total=False):
    text: str
    clean_text: str
    analysis: EmotionResult
    response: str


llm = ChatOpenAI(
    model=settings.model,
    api_key=settings.openai_api_key,
    temperature=0.3,
)

graph = StateGraph(GraphState)


def preprocess_node(state: GraphState) -> Dict:
    clean_text = preprocess_text(state["text"])
    return {"clean_text": clean_text}


def analyze_node(state: GraphState) -> Dict:
    prompt = f"""
    Проанализируй эмоциональное и ментальное состояние человека по сообщению:
    "{state['clean_text']}"
    Верни JSON строго в формате:
    {{
        "emotion": "positive/neutral/negative",
        "risk_level": "low/medium/high",
        "summary": "краткое объяснение"
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
        validated = EmotionResult(**parsed)
    except Exception:
        validated = EmotionResult(
            emotion="neutral",
            risk_level="low",
            summary="Не удалось определить состояние.",
        )
    return {"analysis": validated}


def respond_node(state: GraphState) -> Dict:
    analysis = state["analysis"]
    msg = (
        f"🧠 Эмоциональное состояние: {analysis.emotion}\n"
        f"⚠️ Уровень риска: {analysis.risk_level}\n"
        f"💬 Пояснение: {analysis.summary}"
    )
    return {"response": msg}


# Добавляем узлы
graph.add_node("preprocess_node", preprocess_node)
graph.add_node("analyze_node", analyze_node)
graph.add_node("respond_node", respond_node)

# Устанавливаем точки входа и связи
graph.add_edge(START, "preprocess_node")
graph.add_edge("preprocess_node", "analyze_node")
graph.add_edge("analyze_node", "respond_node")
graph.add_edge("respond_node", END)

compiled_graph = graph.compile()

async def run_graph(text: str) -> str:
    result = await compiled_graph.ainvoke({"text": text})
    return result["response"]
