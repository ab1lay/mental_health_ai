from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from src.core.emotion_schema import EmotionResult
from src.core.text_preprocessing import preprocess_text


class GraphState(TypedDict, total=False):
    text: str
    history: List[str]
    clean_text: str
    language: str
    has_specialist: bool
    preferences: Dict
    analysis: EmotionResult
    response: str


REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return str(model).strip().lower().startswith(REASONING_MODEL_PREFIXES)


def _make_llm(temperature: float) -> ChatOpenAI:
    # Reasoning models reject sampling parameters such as `temperature`
    # (langchain-openai strips it automatically only for gpt-5), so they get
    # `reasoning_effort` instead; classic models keep `temperature`.
    if is_reasoning_model(settings.model):
        tuning = {"reasoning_effort": settings.reasoning_effort}
    else:
        tuning = {"temperature": temperature}
    return ChatOpenAI(model=settings.model, api_key=settings.openai_api_key, **tuning)


# Model for user-facing replies
llm_reply = _make_llm(temperature=0.5)

# Model for structured analysis (JSON)
llm_struct = _make_llm(temperature=0.1).bind(response_format={"type": "json_object"})

graph = StateGraph(GraphState)


def _lang_name(language: str) -> str:
    return {
        "en": "English",
        "ru": "Russian",
        "kk": "Kazakh",
    }.get(language, "English")


def _summary_template(language: str) -> str:
    templates = {
        "en": (
            "**Support Summary**\n\n"
            "**Risk Level:** <High|Medium|Low>\n\n"
            "**Emotional Profile:** <2 sentences describing tone and affect>\n\n"
            "**Status:** <short status>\n\n"
            "**State:** <short state>\n\n"
            "**Requirement:** <short requirement>\n\n"
            "**Notice:** This is a summary for documentation and is not medical advice. "
            "If the situation escalates, contact emergency services or a crisis professional immediately."
        ),
        "ru": (
            "**Сводка поддержки**\n\n"
            "**Уровень риска:** <Высокий|Средний|Низкий>\n\n"
            "**Эмоциональный профиль:** <2 предложения о тоне и состоянии>\n\n"
            "**Статус:** <кратко>\n\n"
            "**Состояние:** <кратко>\n\n"
            "**Потребность:** <кратко>\n\n"
            "**Примечание:** Это сводка для документации и не является медицинской консультацией. "
            "Если ситуация ухудшается, немедленно обратитесь в экстренные службы или к кризисному специалисту."
        ),
        "kk": (
            "**Қолдау қорытындысы**\n\n"
            "**Тәуекел деңгейі:** <Жоғары|Орта|Төмен>\n\n"
            "**Эмоциялық профиль:** <тона мен күйі туралы 2 сөйлем>\n\n"
            "**Мәртебе:** <қысқа>\n\n"
            "**Күйі:** <қысқа>\n\n"
            "**Қажет:** <қысқа>\n\n"
            "**Ескерту:** Бұл құжаттандыруға арналған қысқаша мәлімет және медициналық кеңес емес. "
            "Жағдай ушықса, дереу жедел қызметтерге немесе дағдарыс маманына хабарласыңыз."
        ),
    }
    return templates.get(language, templates["en"])


def _summary_title(language: str, risk_level: str) -> str:
    high_titles = {
        "en": "**🚨 Urgent Support Alert**",
        "ru": "**🚨 Срочный сигнал поддержки**",
        "kk": "**🚨 Шұғыл қолдау туралы хабарлама**",
    }
    support_titles = {
        "en": "**Support Summary**",
        "ru": "**Сводка поддержки**",
        "kk": "**Қолдау қорытындысы**",
    }
    titles = high_titles if str(risk_level).lower() == "high" else support_titles
    return titles.get(language, titles["en"])


def _apply_summary_title(summary: str, language: str, risk_level: str) -> str:
    lines = (summary or "").splitlines()
    title = _summary_title(language, risk_level)
    known_title_markers = (
        "Urgent Support Alert",
        "Support Summary",
        "Срочный сигнал поддержки",
        "Сводка поддержки",
        "Шұғыл қолдау",
        "Қолдау қорытындысы",
    )
    for index, line in enumerate(lines):
        if line.strip():
            if any(marker in line for marker in known_title_markers):
                lines[index] = title
            else:
                lines.insert(index, title)
            return "\n".join(lines).strip()
    return title


def _fallback_user_reply(language: str) -> str:
    return {
        "en": "I hear you. That sounds really heavy. You are not alone. What feels most difficult right now?",
        "ru": "Я слышу тебя. Это и правда очень тяжело. Ты не один(а). Что сейчас беспокоит больше всего?",
        "kk": "Мен сені естіп тұрмын. Бұл шындығында қиын. Сен жалғыз емессің. Қазір ең мазалайтын нәрсе не?",
    }.get(language, "I hear you. That sounds really heavy. You are not alone. What feels most difficult right now?")


def _fallback_proactive_checkin(language: str) -> str:
    return {
        "en": "You matter, and I am glad to check in. How are you feeling today?",
        "ru": "Вы важны, и я решил(а) написать вам. Как вы себя чувствуете сегодня?",
        "kk": "Сіз маңыздысыз, мен сізге хал-жағдайыңызды сұрау үшін жаздым. Бүгін өзіңізді қалай сезінесіз?",
    }.get(language, "You matter, and I am glad to check in. How are you feeling today?")


def preprocess_node(state: GraphState) -> Dict:
    clean_text = preprocess_text(state["text"])
    history = state.get("history", [])
    language = state.get("language", "en")
    has_specialist = state.get("has_specialist", False)
    preferences = state.get("preferences") or {}
    return {
        "clean_text": clean_text,
        "history": history,
        "language": language,
        "has_specialist": has_specialist,
        "preferences": preferences,
    }


def _preferences_prompt_block(preferences: Dict) -> str:
    if not preferences:
        return ""
    tone = preferences.get("tone", "supportive")
    response_length = preferences.get("response_length", "medium")
    support_style = preferences.get("support_style", "reflective")
    topics_to_avoid = (preferences.get("topics_to_avoid") or "").strip()
    lines = [
        "Patient personalization preferences:",
        f"- preferred tone: {tone}",
        f"- preferred response length: {response_length}",
        f"- preferred support style: {support_style}",
    ]
    if topics_to_avoid:
        lines.append(f"- avoid these topics unless user explicitly asks: {topics_to_avoid}")
    return "\n".join(lines) + "\n"


def _normalize_parsed_payload(parsed: dict) -> dict:
    characteristics = parsed.get("key_characteristics")
    if isinstance(characteristics, str):
        parsed["key_characteristics"] = [characteristics]
    elif not isinstance(characteristics, list):
        parsed["key_characteristics"] = []
    return parsed


def _combined_signal_text(text: str, history: List[str] | None = None) -> str:
    chunks = [text or ""]
    # Only the patient's own words count as signals: assistant replies such as
    # "you are not alone" must not raise the patient's risk.
    for line in history or []:
        line = line or ""
        if line.startswith("assistant:"):
            continue
        if line.startswith("user:"):
            line = line[len("user:"):]
        chunks.append(line)
    return " ".join(chunks).lower()


def _has_any_signal(text: str, signals: tuple[str, ...]) -> bool:
    return any(signal in text for signal in signals)


# Explicit self-harm / suicidal ideation. Matching any of these forces high risk
# regardless of what the model returned (and even when the model call failed).
_CRISIS_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:i\s+)?want\s+to\s+die\b",
        r"\b(?:kill|hurt|harm|cut)\s+myself\b",
        r"\bend\s+(?:my|it)\s+(?:life|all)\b",
        r"\btake\s+my\s+(?:own\s+)?life\b",
        r"\b(?:don'?t|do\s+not)\s+want\s+to\s+(?:live|be\s+alive|wake\s+up)\b",
        r"\b(?:no|zero)\s+reason\s+to\s+live\b",
        r"\bsuicid",
        r"\bself[-\s]?harm",
        r"\bbetter\s+off\s+dead\b",
        r"не\s+хоч\w*\s+(?:больше\s+)?жить",
        r"хоч\w*\s+умереть",
        r"поконч\w*\s+(?:с\s+собой|со\s+всем)",
        r"(?:убить|убью)\s+себя",
        r"суицид",
        r"самоубийств",
        r"(?:причин\w*|наносить|нанести)\s+себе\s+вред",
        r"навредить\s+себе",
        r"порезать\s+себя|режу\s+себя",
        r"нет\s+смысла\s+жить",
        r"лучше\s+бы\s+меня\s+не\s+было",
        r"өмір\s+сүргім\s+келмейді",
        r"өлгім\s+келеді",
        r"өзімді\s+өлтір",
        r"өзіме\s+қол\s+жұмса",
        r"өмірдің\s+мәні\s+жоқ",
    )
)


def has_crisis_signal(text: str) -> bool:
    lowered = (text or "").lower().replace("’", "'")
    return any(pattern.search(lowered) for pattern in _CRISIS_PATTERNS)


def _risk_rank(risk_level: str) -> int:
    return {"low": 1, "medium": 2, "medium_high": 3, "high": 4}.get(
        str(risk_level).strip().lower(),
        1,
    )


def _clamp_score(value) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_analysis_values(analysis: EmotionResult) -> None:
    """Coerce model output ("High", "medium-high", 1.3) to the expected vocabulary."""
    risk = str(analysis.risk_level or "").strip().lower().replace("-", "_").replace(" ", "_")
    analysis.risk_level = risk if risk in {"low", "medium", "medium_high", "high"} else "low"
    emotion = str(analysis.emotion or "").strip().lower()
    analysis.emotion = emotion if emotion in {"positive", "neutral", "negative"} else "neutral"
    analysis.depression_score = _clamp_score(analysis.depression_score)
    analysis.anxiety_score = _clamp_score(analysis.anxiety_score)
    analysis.stress_score = _clamp_score(analysis.stress_score)


def _raise_risk_at_least(analysis: EmotionResult, minimum: str) -> None:
    if _risk_rank(analysis.risk_level) < _risk_rank(minimum):
        analysis.risk_level = minimum


def should_create_alert(analysis: EmotionResult) -> bool:
    risk_level = str(analysis.risk_level).strip().lower()
    if risk_level == "high":
        return True
    if (
        float(analysis.depression_score) >= 0.7
        and risk_level in {"medium", "medium_high"}
    ):
        return True
    return False


def _calibrate_analysis(
    analysis: EmotionResult,
    text: str,
    language: str,
    history: List[str] | None = None,
) -> EmotionResult:
    """Deterministic safety calibration for known under-classified patterns."""
    _normalize_analysis_values(analysis)
    signal_text = _combined_signal_text(text, history)

    # Short ambiguous tokens ("один", "одна", "alone") are deliberately not used:
    # they match "один раз", "однажды", "leave me alone" and caused false alerts.
    social_withdrawal_signals = (
        "withdraw", "isolat", "avoid everyone", "avoid people", "no one cares",
        "no one understands", "feel alone", "all alone", "so alone",
        "completely alone", "lonely", "emptiness", "empty inside", "feel empty",
        "hopeless", "no hope", "pointless", "nothing matters",
        "изолир", "замкнул", "избега", "одинок", "одиночеств", "совсем один",
        "совсем одна", "никому не нужн", "пустот", "внутри пусто", "на душе пусто",
        "безнадеж", "безнадёж", "нет надежды", "нет смысла",
        "бессмысленно", "ничего не важно",
        "жалғыз", "оқшау", "үміт жоқ", "үмітсіз", "бос сияқты", "мағына жоқ",
    )
    physical_anxiety_signals = (
        "hands shaking", "hand shaking", "shaky hands", "shaking before",
        "heart racing", "racing heart", "can't breathe", "cant breathe", "cannot breathe",
        "shortness of breath", "tight chest", "chest tight", "panic",
        "дрожат руки", "руки тряс", "сердце колот", "сердцеби", "не могу дыш",
        "тяжело дыш", "сдавило груд", "паник",
        "қолым діріл", "қолдарым діріл", "жүрегім қатты", "дем ала алмай",
        "тыныс алу қиын", "кеудем қыс", "үрей",
    )

    if _has_any_signal(signal_text, social_withdrawal_signals):
        analysis.depression_score = max(float(analysis.depression_score), 0.7)
        _raise_risk_at_least(analysis, "medium")

    if _has_any_signal(signal_text, physical_anxiety_signals):
        analysis.anxiety_score = max(float(analysis.anxiety_score), 0.7)
        _raise_risk_at_least(analysis, "medium")

    # Crisis check looks at the current text only, so one past message does not
    # keep every later reply in crisis mode.
    if has_crisis_signal(text):
        analysis.depression_score = max(float(analysis.depression_score), 0.8)
        _raise_risk_at_least(analysis, "high")

    analysis.summary = _apply_summary_title(
        analysis.summary,
        language=language,
        risk_level=analysis.risk_level,
    )
    return analysis


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

    banned_phrases = (
        "tone only", "tone:", "overall tone", "neutral tone", "communication is neutral",
        "\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043d\u043e\u0441\u0438\u0442 \u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440",
        "\u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0442\u043e\u043d",
        "\u043e\u0431\u0449\u0438\u0439 \u0442\u043e\u043d",
    )
    if any(phrase in lowered for phrase in banned_phrases):
        return True

    meta_phrases = (
        "neutral greeting",
        "initiates conversation",
        "expresses neutral mood",
        "seeks emotional connection",
        "starts conversation",
        "\u043d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u043e\u0435 \u043f\u0440\u0438\u0432\u0435\u0442\u0441\u0442\u0432\u0438\u0435",
        "\u0438\u043d\u0438\u0446\u0438\u0438\u0440\u0443\u0435\u0442 \u0434\u0438\u0430\u043b\u043e\u0433",
    )
    return any(phrase in lowered for phrase in meta_phrases)


def _is_specific_characteristic(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    markers = (
        "\u0438\u0437\u0431\u0435\u0433", "\u0442\u0440\u0443\u0434\u043d\u043e", "\u0441\u043b\u043e\u0436\u043d\u043e",
        "\u0442\u0440\u0435\u0432\u043e\u0433", "\u0430\u043d\u0438\u043c\u0435", "\u043c\u0430\u043d\u0433\u0430",
        "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436", "\u0436\u0430\u043d\u0440", "\u043c\u0443\u0437\u044b\u043a",
        "\u0438\u043d\u0442\u0435\u0440\u0435\u0441", "\u043b\u044e\u0431\u0438\u043c", "\u043d\u0440\u0430\u0432",
        "\u0442\u043e\u043b\u043f", "\u0448\u0443\u043c", "\u0442\u0440\u0438\u0433\u0433\u0435\u0440",
        "\u0441\u043a\u0443\u0447\u0430\u0435\u0442", "\u043e\u0434\u0438\u043d\u043e\u043a", "\u0441\u043e\u043d",
        "struggle", "avoids", "worse", "better", "sleep", "appetite", "anxiety", "panic",
        "crowd", "noise", "trigger", "prefers", "interested", "likes", "favorite", "hobby",
        "anime", "manga", "character", "genre", "music", "movie", "book",
    )
    return any(m in lowered for m in markers)


def _extract_fallback_characteristics_from_messages(messages: List[str], language: str) -> List[str]:
    joined = " ".join((m or "") for m in messages).lower()
    items: List[str] = []

    if any(k in joined for k in ["anime", "manga", "\u0430\u043d\u0438\u043c\u0435", "\u043c\u0430\u043d\u0433\u0430"]):
        if any(k in joined for k in ["favorite", "likes", "\u043b\u044e\u0431\u0438\u043c", "\u043d\u0440\u0430\u0432"]):
            items.append(
                "\u041f\u0440\u043e\u044f\u0432\u043b\u044f\u0435\u0442 \u0443\u0441\u0442\u043e\u0439\u0447\u0438\u0432\u044b\u0439 \u0438\u043d\u0442\u0435\u0440\u0435\u0441 \u043a \u0430\u043d\u0438\u043c\u0435 \u0438 \u043b\u044e\u0431\u0438\u043c\u044b\u043c \u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436\u0430\u043c."
                if language == "ru"
                else "Shows a stable interest in anime and favorite characters."
            )
        else:
            items.append(
                "\u0418\u043d\u0442\u0435\u0440\u0435\u0441\u0443\u0435\u0442\u0441\u044f \u0430\u043d\u0438\u043c\u0435."
                if language == "ru"
                else "Interested in anime."
            )

    if any(k in joined for k in ["music", "genre", "\u043c\u0443\u0437\u044b\u043a", "\u0436\u0430\u043d\u0440"]):
        items.append(
            "\u041e\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u043c\u0443\u0437\u044b\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0442\u0435\u043d\u0438\u044f \u0438 \u043b\u044e\u0431\u0438\u043c\u044b\u0435 \u0436\u0430\u043d\u0440\u044b."
            if language == "ru"
            else "Mentions genre or music preferences."
        )

    if (
        any(k in joined for k in ["crowd", "noise", "\u0442\u043e\u043b\u043f", "\u0448\u0443\u043c"])
        and any(k in joined for k in ["uncomfortable", "hard", "\u0434\u0438\u0441\u043a\u043e\u043c\u0444\u043e\u0440\u0442", "\u0442\u044f\u0436\u0435\u043b"])
    ):
        items.append(
            "\u0418\u0441\u043f\u044b\u0442\u044b\u0432\u0430\u0435\u0442 \u0434\u0438\u0441\u043a\u043e\u043c\u0444\u043e\u0440\u0442 \u0432 \u0442\u043e\u043b\u043f\u0435 \u0438 \u0448\u0443\u043c\u043d\u044b\u0445 \u043c\u0435\u0441\u0442\u0430\u0445."
            if language == "ru"
            else "Feels discomfort in crowds and noisy places."
        )

    # keep only specific-ish fallback entries
    cleaned = [i for i in items if not _is_generic_characteristic(i)]
    return cleaned[:3]


def _sanitize_characteristics(raw_items: List[str], messages: List[str] | None = None, language: str = "en") -> List[str]:
    specific: List[str] = []
    fallback_general: List[str] = []
    generic_rejected: List[str] = []
    rejected_details: List[Dict[str, str]] = []
    seen = set()
    for raw in raw_items:
        if not isinstance(raw, str):
            continue
        parts = [p.strip(" -	?") for p in raw.replace(";", ",").split(",")]
        for part in parts:
            if len(part) < 4:
                rejected_details.append({"value": part[:160], "reason": "too_short"})
                continue
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            if _is_generic_characteristic(part):
                generic_rejected.append(part[:160])
                rejected_details.append({"value": part[:160], "reason": "generic"})
                continue
            if _is_specific_characteristic(part):
                specific.append(part[:160])
            else:
                fallback_general.append(part[:160])
                rejected_details.append({"value": part[:160], "reason": "non_specific_fallback"})

    if specific:
        logging.debug(
            "Characteristics sanitize: specific=%s fallback=%s generic_rejected=%s selected=%s",
            len(specific),
            len(fallback_general),
            len(generic_rejected),
            specific[:3],
        )
        return specific[:8]

    # If model failed to produce concrete traits, derive lightweight specifics from user messages.
    heuristic = _extract_fallback_characteristics_from_messages(messages or [], language=language)
    if heuristic:
        logging.debug(
            "Characteristics sanitize: no specific from model, using heuristic fallback=%s generic_rejected=%s",
            heuristic[:3],
            len(generic_rejected),
        )
        return heuristic[:3]

    # Only if absolutely nothing concrete found, keep one non-generic fallback.
    selected = fallback_general[:1]
    logging.debug(
        "Characteristics sanitize: selected fallback=%s generic_rejected=%s rejected_details=%s",
        selected,
        len(generic_rejected),
        rejected_details[:5],
    )
    return selected


def _parse_emotion_result(content: str) -> EmotionResult:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[len("json") :].lstrip("\n")
    return EmotionResult(**_normalize_parsed_payload(json.loads(raw)))


def _fallback_analysis(user_reply: str) -> EmotionResult:
    return EmotionResult(
        emotion="neutral",
        risk_level="low",
        depression_score=0.0,
        anxiety_score=0.0,
        stress_score=0.0,
        summary="Automatic analysis is unavailable for this fragment; please review the dialog manually.",
        profile_note="analysis unavailable",
        user_reply=user_reply,
        key_characteristics=[],
    )


def analyze_node(state: GraphState) -> Dict:
    history_block = ""
    history = state.get("history") or []
    language = state.get("language", "en")
    has_specialist = state.get("has_specialist", False)
    preferences = state.get("preferences") or {}
    if history:
        joined_history = "\n".join([f"- {msg}" for msg in history[-8:]])
        history_block = (
            f"Recent dialog messages (role:text):\n{joined_history}\n\n"
        )

    specialist_note = ""
    if has_specialist:
        specialist_note = (
            "The patient is already linked to a specialist. "
            "Do not suggest finding a specialist. "
            "Instead, suggest contacting their specialist or say you can notify them."
        )

    prompt = f"""
You are a supportive conversational partner. Based on the latest message and context, produce a warm, natural reply in first person.
Use {_lang_name(language)} for all user-facing text.
In summaries, use "Patient" (or "Пациент") instead of "User"/"Пользователь".
Summary title rule: use "Support Summary" for low/medium risk and "Urgent Support Alert" only for high risk.
Tone: empathetic, reflective, grounded. Mirror feelings, summarize what you heard, then offer one or two gentle next steps or questions.
Length: no hard limit, but stay concise and helpful (2 short paragraphs max).
Directness: medium - clear and caring, not clinical.
Safety: if there are signals of self-harm or acute crisis, ask a brief safety check question first. Offer to share emergency resources after the user responds.
Do NOT repeatedly recommend contacting a specialist in every reply. Only mention contacting a specialist if risk is high or the user explicitly asks for help.
{specialist_note}
{_preferences_prompt_block(preferences)}
Return a JSON object following this schema (JSON only, no extra text):
{{
    "emotion": "positive|neutral|negative",
    "risk_level": "low|medium|high",
    "depression_score": 0.0-1.0,
    "anxiety_score": 0.0-1.0,
    "stress_score": 0.0-1.0,
    "summary": "Return ONLY markdown in this exact format (no prefacing text), matching the style below. Preserve the **bold** markers:\n\n{_summary_template(language)}",
    "profile_note": "2-3 words about tone/needs/resources",
    "user_reply": "plain response text only (no labels, no meta), empathetic with reflection + gentle next step",
    "key_characteristics": [
      "3-6 concrete observations useful for specialist profile, inferred from dialog only",
      "Use specific patterns/triggers/impairments/resources, e.g. 'feels discomfort in crowds', 'avoids noisy places', 'sleep worsens on stressful days'",
      "Do NOT include generic tone adjectives like 'positive', 'engaged', 'calm'",
      "If no concrete characteristics can be inferred, return one fallback general characteristic only"
    ]
}}
{history_block}
Message: "{state["clean_text"]}"
Return JSON only. Do not add any extra text outside the JSON object.
"""
    try:
        response = llm_struct.invoke([HumanMessage(content=prompt)])
        validated = _parse_emotion_result(response.content)
        raw_characteristics = list(validated.key_characteristics or [])
        validated.key_characteristics = _sanitize_characteristics(validated.key_characteristics, messages=history, language=language)
        logging.debug(
            "Characteristics analyze_node: raw=%s sanitized=%s",
            raw_characteristics,
            validated.key_characteristics,
        )
    except Exception:
        logging.exception("Emotion analysis failed, using fallback result")
        validated = _fallback_analysis(_fallback_user_reply(language))
    # Calibration also runs on the fallback so explicit crisis signals are
    # never lost when the model call or JSON parsing fails.
    validated = _calibrate_analysis(
        validated,
        text=state["clean_text"],
        language=language,
        history=history,
    )
    # Guardrail: if model returns placeholder/meta instead of a real reply.
    if validated.user_reply and any(
        token in validated.user_reply.lower()
        for token in [
            "empathetic response",
            "gentle next step",
            "medium directness",
            "user_reply",
        ]
    ):
        validated.user_reply = _fallback_user_reply(language)
    return {"analysis": validated}


def respond_node(state: GraphState) -> Dict:
    analysis = state["analysis"]
    if analysis.user_reply:
        return {"response": analysis.user_reply}

    # Fallback if model did not return user_reply
    return {"response": _fallback_user_reply(state.get("language", "en"))}


graph.add_node("preprocess_node", preprocess_node)
graph.add_node("analyze_node", analyze_node)
graph.add_node("respond_node", respond_node)

graph.add_edge(START, "preprocess_node")
graph.add_edge("preprocess_node", "analyze_node")
graph.add_edge("analyze_node", "respond_node")
graph.add_edge("respond_node", END)

compiled_graph = graph.compile()


async def run_graph(text: str, history: List[str] | None = None, language: str = "en") -> str:
    result = await compiled_graph.ainvoke({"text": text, "history": history or [], "language": language})
    return result["response"]


async def run_graph_with_analysis(
    text: str,
    history: List[str] | None = None,
    language: str = "en",
    has_specialist: bool = False,
    preferences: Dict | None = None,
) -> tuple[str, EmotionResult]:
    """Returns the user-facing reply together with the per-message analysis."""
    result = await compiled_graph.ainvoke(
        {
            "text": text,
            "history": history or [],
            "language": language,
            "has_specialist": has_specialist,
            "preferences": preferences or {},
        }
    )
    return result["response"], result["analysis"]


async def run_graph_with_context(
    text: str,
    history: List[str] | None = None,
    language: str = "en",
    has_specialist: bool = False,
    preferences: Dict | None = None,
) -> str:
    response, _ = await run_graph_with_analysis(
        text,
        history=history,
        language=language,
        has_specialist=has_specialist,
        preferences=preferences,
    )
    return response


async def analyze_dialog(messages: List[str], language: str = "en") -> EmotionResult:
    """Periodic analysis over a list of user-only messages."""
    formatted = "\n".join([f"- {m}" for m in messages[-10:]]) if messages else "no messages"
    prompt = f"""
You are a psychological assistant. Analyze the dialog (user-only messages) and return JSON with this schema.
Use {_lang_name(language)} for all user-facing text.
In summaries, use "Patient" (or "Пациент") instead of "User"/"Пользователь".
Summary title rule: use "Support Summary" for low/medium risk and "Urgent Support Alert" only for high risk.
{{
    "emotion": "positive|neutral|negative",
    "risk_level": "low|medium|high",
    "depression_score": 0.0-1.0,
    "anxiety_score": 0.0-1.0,
    "stress_score": 0.0-1.0,
    "summary": "Return ONLY markdown in this exact format (no prefacing text), matching the style below. Preserve the **bold** markers:\n\n{_summary_template(language)}",
    "profile_note": "2-3 words about tone/needs/resources",
    "user_reply": "brief warm support (optional for report)",
    "key_characteristics": [
      "3-6 concrete observations useful for specialist profile, inferred from dialog only",
      "Use specific patterns/triggers/impairments/resources, e.g. 'feels discomfort in crowds', 'avoids noisy places', 'sleep worsens on stressful days'",
      "Do NOT include generic tone adjectives like 'positive', 'engaged', 'calm'",
      "If no concrete characteristics can be inferred, return one fallback general characteristic only"
    ]
}}
User messages:
{formatted}
"""
    try:
        response = await llm_struct.ainvoke([HumanMessage(content=prompt)])
        result = _parse_emotion_result(response.content)
        raw_characteristics = list(result.key_characteristics or [])
        result.key_characteristics = _sanitize_characteristics(result.key_characteristics, messages=messages, language=language)
        logging.debug(
            "Characteristics analyze_dialog: raw=%s sanitized=%s",
            raw_characteristics,
            result.key_characteristics,
        )
    except Exception:
        logging.exception("Dialog analysis failed, using fallback result")
        result = _fallback_analysis("")
    return _calibrate_analysis(
        result,
        text="\n".join(messages),
        language=language,
        history=messages,
    )


async def summarize_dialog(messages: List[str], language: str = "en") -> tuple[EmotionResult, str]:
    """Creates a short summary from recent user messages."""
    analysis = await analyze_dialog(messages, language=language)
    lines = [
        "Dialog summary:",
        f"- Emotion/tone: {analysis.emotion}",
        f"- Risk: {analysis.risk_level}",
        f"- Summary: {analysis.summary}",
        f"- Topics/needs: {analysis.profile_note or 'unknown'}",
        "- Reminder: this is not medical advice. If things worsen, contact a professional or hotline.",
    ]
    return analysis, "\n".join(lines)


async def generate_proactive_checkin(
    history: List[str] | None = None,
    language: str = "en",
    preferences: Dict | None = None,
) -> str:
    history_lines = history or []
    history_block = "\n".join(f"- {line}" for line in history_lines[-8:])
    preferences = preferences or {}
    tone = preferences.get("tone", "supportive")
    response_length = preferences.get("response_length", "medium")
    support_style = preferences.get("support_style", "reflective")
    topics_to_avoid = (preferences.get("topics_to_avoid") or "").strip()
    prompt = f"""
Create one short proactive check-in message for a patient who has been silent for a while.
Language: {_lang_name(language)}.
Rules:
- Start with a positive affirmation or warm greeting.
- If context exists, lightly reference it without sounding intrusive.
- Ask how the person is doing now.
- Keep it to 1-2 sentences, natural and supportive.
- Do not mention policy, diagnosis, or medical claims.
- Follow patient preferences:
  - tone: {tone}
  - response_length: {response_length}
  - support_style: {support_style}
  - topics to avoid: {topics_to_avoid or 'none'}

Recent context (if any):
{history_block if history_block else "- no recent context"}
"""
    try:
        response = await llm_reply.ainvoke([HumanMessage(content=prompt)])
        text = (response.content or "").strip()
        return text or _fallback_proactive_checkin(language)
    except Exception:
        logging.exception("Failed to generate proactive check-in")
        return _fallback_proactive_checkin(language)
