import os
import re
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
import csv
import io
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
import markdown as md
from dotenv import load_dotenv

from .auth import decode_session, encode_session, hash_password, verify_password
from .db import (
    ensure_tables,
    sync_patients_from_messages,
    get_user_by_email,
    get_user_by_id,
    create_user,
    set_user_report_language,
    get_user_report_language,
    list_patients_for_user,
    get_patient,
    get_patient_language,
    get_patient_preferences,
    list_patient_preference_history,
    list_summaries,
    list_messages,
    list_patient_characteristics,
    list_alerts,
    resolve_alert,
    add_audit_event,
    get_last_summary,
    get_patient_by_chat_id,
    count_patients,
    count_active_alerts,
    create_invite,
    list_invites,
    clear_summaries_for_chat,
    clear_messages_for_chat,
    clear_timeline_for_chat,
    list_audit_events,
    list_timeline,
    upsert_patient_preferences,
    consume_patient_portal_token,
)

BASE_DIR = Path(__file__).resolve().parents[1]
# Prefer local env in healthcare_web/, fallback to project root.
local_env = BASE_DIR / '.env'
root_env = BASE_DIR.parent / '.env'
if local_env.exists():
    load_dotenv(local_env)
else:
    load_dotenv(root_env)

SECRET_KEY = os.getenv('WEB_SECRET_KEY', 'change-me')
SESSION_TTL_SECONDS = int(os.getenv('WEB_SESSION_TTL_SECONDS', '86400'))
PATIENT_SESSION_TTL_SECONDS = int(os.getenv('PATIENT_WEB_SESSION_TTL_SECONDS', '604800'))
DEBUG_ENABLED = os.getenv('DEBUG', 'false').lower() in ('1', 'true', 'yes', 'on')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

app = FastAPI(title='Healthcare Web')
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'app' / 'static')), name='static')

templates = Jinja2Templates(directory=str(BASE_DIR / 'app' / 'templates'))


def _patient_texts(language: str) -> dict:
    if language == "ru":
        return {
            "portal_title": "Портал пациента",
            "logout": "Выход",
            "page_title": "Персонализация бота",
            "page_subtitle": "Настройте, как бот общается и как часто может сам писать вам.",
            "chat_id": "Chat ID",
            "saved": "Настройки сохранены.",
            "tone": "Тон",
            "tone_tip": "Задает стиль общения. Поддерживающий = больше эмпатии, Спокойный = нейтральнее и ровнее, Прямой = короче и по делу.",
            "response_length": "Длина ответа",
            "response_length_tip": "Коротко = краткие ответы, Средне = сбалансированно, Подробно = больше деталей и объяснений.",
            "support_style": "Формат поддержки",
            "support_style_tip": "Рефлексивный = больше валидации и отражения, Практический = конкретные шаги, Мотивирующий = больше подбадривания.",
            "proactive_frequency": "Частота инициативных сообщений (часы)",
            "proactive_frequency_tip": "Минимальная пауза перед тем, как бот снова может написать первым.",
            "quiet_start": "Тихие часы: начало (HH:MM)",
            "quiet_start_tip": "Начало периода, когда бот не должен писать первым. Формат 24ч: HH:MM.",
            "quiet_end": "Тихие часы: конец (HH:MM)",
            "quiet_end_tip": "Конец тихого периода. Пример: 22:00–08:00 блокирует ночные check-in.",
            "topics_to_avoid": "Темы, которых избегать (опционально)",
            "topics_to_avoid_tip": "Темы, которые бот не поднимает сам, если вы явно не попросите.",
            "topics_placeholder": "Например: не обсуждать семью в инициативных сообщениях",
            "allow_proactive": "Разрешить боту писать первым",
            "allow_proactive_tip": "Если выключено, бот только отвечает на ваши сообщения и не начинает диалог сам.",
            "save": "Сохранить настройки",
            "history_title": "История изменений",
            "history_empty": "Пока нет изменений.",
            "history_who": "Кто",
            "history_when": "Когда",
            "history_what": "Что изменено",
            "history_actor_patient": "Пациент",
            "history_actor_specialist": "Специалист",
            "history_actor_system": "Система",
            "tone_supportive": "Поддерживающий",
            "tone_calm": "Спокойный",
            "tone_direct": "Прямой",
            "len_short": "Коротко",
            "len_medium": "Средне",
            "len_detailed": "Подробно",
            "style_reflective": "Рефлексивный",
            "style_practical": "Практический",
            "style_motivational": "Мотивирующий",
        }
    return {
        "portal_title": "Patient Portal",
        "logout": "Logout",
        "page_title": "Bot Personalization",
        "page_subtitle": "Adjust how your bot speaks and when it can proactively check in.",
        "chat_id": "Chat ID",
        "saved": "Preferences saved.",
        "tone": "Tone",
        "tone_tip": "Sets communication style. Supportive = warmer empathy, Calm = steadier/neutral wording, Direct = more concise and straightforward.",
        "response_length": "Response length",
        "response_length_tip": "Controls how long bot replies are. Short = brief, Medium = balanced, Detailed = more context and explanations.",
        "support_style": "Support style",
        "support_style_tip": "Reflective = more validation and mirroring, Practical = concrete steps, Motivational = encouraging framing.",
        "proactive_frequency": "Proactive frequency (hours)",
        "proactive_frequency_tip": "Minimum pause before bot can initiate a check-in after inactivity.",
        "quiet_start": "Quiet hours start (HH:MM)",
        "quiet_start_tip": "Start of time window when bot should not send proactive messages. 24-hour HH:MM.",
        "quiet_end": "Quiet hours end (HH:MM)",
        "quiet_end_tip": "End of quiet window. Example: 22:00 to 08:00 blocks nighttime proactive pings.",
        "topics_to_avoid": "Topics to avoid (optional)",
        "topics_to_avoid_tip": "Topics bot should avoid unless you explicitly ask.",
        "topics_placeholder": "Example: avoid discussing family in proactive messages",
        "allow_proactive": "Allow bot to proactively message me",
        "allow_proactive_tip": "If disabled, bot will never start conversation first; it will only reply when you write.",
        "save": "Save preferences",
        "history_title": "Change History",
        "history_empty": "No changes yet.",
        "history_who": "Who",
        "history_when": "When",
        "history_what": "What changed",
        "history_actor_patient": "Patient",
        "history_actor_specialist": "Specialist",
        "history_actor_system": "System",
        "tone_supportive": "Supportive",
        "tone_calm": "Calm",
        "tone_direct": "Direct",
        "len_short": "Short",
        "len_medium": "Medium",
        "len_detailed": "Detailed",
        "style_reflective": "Reflective",
        "style_practical": "Practical",
        "style_motivational": "Motivational",
    }

def _cleanup_summary(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("Markdown in this exact format:", "").strip()
    cleaned = cleaned.replace("Return ONLY markdown in this exact format (no prefacing text):", "").strip()
    return _normalize_summary(cleaned)


def _normalize_summary(text: str) -> str:
    if not text:
        return text
    # Remove specific broken bold patterns without nuking valid markdown.
    text = text.replace("** Status: **", "**Status:**")
    text = text.replace("** State: **", "**State:**")
    text = text.replace("** Requirement: **", "**Requirement:**")
    text = text.replace("** Notice: **", "**Notice:**")
    text = text.replace("* *", "")
    # Normalize "Core Situation - Status: ... - State: ... - Requirement: ... - Notice: ..."
    if "Core Situation" in text and "Status:" in text:
        part = text.replace("Core Situation", "").strip()
        for label in ["Status:", "State:", "Requirement:", "Notice:"]:
            part = part.replace(f" - {label}", f"\n**{label}** ")
            part = part.replace(f"- {label}", f"\n**{label}** ")
            part = part.replace(f"{label}", f"\n**{label}** ")
        text = part.strip()

    # Ensure labels appear on new lines even if model outputs them in one line.
    for raw_label in ["Status:", "State:", "Requirement:", "Notice:"]:
        text = text.replace(f" {raw_label}", f"\n{raw_label}")

    # If labels are still on the same line, force breaks by converting them to bold labels.
    if "Status:" in text or "State:" in text or "Requirement:" in text or "Notice:" in text:
        text = text.replace("Status:", "\n\n**Status:**")
        text = text.replace("State:", "\n\n**State:**")
        text = text.replace("Requirement:", "\n\n**Requirement:**")
        text = text.replace("Notice:", "\n\n**Notice:**")

    # Fix accidental spacing in bold markers like "** Status: **"
    text = text.replace("** Status: **", "**Status:**")
    text = text.replace("** State: **", "**State:**")
    text = text.replace("** Requirement: **", "**Requirement:**")
    text = text.replace("** Notice: **", "**Notice:**")

    # Ensure each label starts on its own line (separate paragraph).
    for label in ["**Status:**", "**State:**", "**Requirement:**", "**Notice:**"]:
        text = text.replace(f" {label}", f"\n\n{label}")

    return text.strip()


def _normalize_characteristic_key(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    lowered = re.sub(r"[^\w\s\u0400-\u04FF-]", " ", lowered)

    # Collapse frequent semantic duplicates to one canonical key.
    if any(k in lowered for k in ("anime", "manga", "\u0430\u043d\u0438\u043c\u0435", "\u043c\u0430\u043d\u0433\u0430")):
        if any(k in lowered for k in ("character", "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436")):
            return "interest_anime_characters"
        return "interest_anime"
    if any(
        k in lowered
        for k in (
            "music", "genre", "rock", "rap", "hiphop", "hip-hop",
            "\u043c\u0443\u0437\u044b\u043a", "\u0436\u0430\u043d\u0440", "\u0440\u043e\u043a", "\u0440\u044d\u043f", "\u0445\u0438\u043f\u0445\u043e\u043f", "\u0430\u043b\u044c\u0442\u0435\u0440\u043d\u0430\u0442\u0438\u0432",
        )
    ):
        return "interest_music"
    if any(k in lowered for k in ("character", "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436")):
        return "interest_characters"
    if any(k in lowered for k in ("crowd", "noise", "trigger", "avoid", "\u0442\u043e\u043b\u043f", "\u0448\u0443\u043c", "\u0442\u0440\u0438\u0433\u0433\u0435\u0440", "\u0438\u0437\u0431\u0435\u0433", "\u0434\u0438\u0441\u043a\u043e\u043c\u0444\u043e\u0440\u0442")):
        return "trigger_environment"
    if any(k in lowered for k in ("mood", "emotion", "lonely", "misses", "\u0441\u043a\u0443\u0447", "\u043e\u0434\u0438\u043d\u043e\u043a", "\u0433\u0440\u0443\u0441\u0442", "\u044d\u043c\u043e\u0446")):
        return "emotional_state"

    lowered = lowered.replace("\u043f\u0440\u043e\u044f\u0432\u043b\u044f\u0435\u0442 \u0438\u043d\u0442\u0435\u0440\u0435\u0441 \u043a", "\u0438\u043d\u0442\u0435\u0440\u0435\u0441")
    lowered = lowered.replace("\u0438\u043d\u0442\u0435\u0440\u0435\u0441\u0443\u0435\u0442\u0441\u044f", "\u0438\u043d\u0442\u0435\u0440\u0435\u0441")
    lowered = lowered.replace("\u043b\u044e\u0431\u0438\u0442", "\u043b\u044e\u0431\u0438\u043c")
    lowered = lowered.replace("\u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f", "\u043d\u0440\u0430\u0432")
    lowered = lowered.replace("\u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0438\u0442\u0430\u0435\u0442", "\u043f\u0440\u0435\u0434\u043f\u043e\u0447")
    lowered = lowered.replace("interested in", "interest")
    lowered = lowered.replace("likes", "like")
    stopwords = {
        "\u0438", "\u0432", "\u0432\u043e", "\u043d\u0430", "\u043f\u043e", "\u0441", "\u0441\u043e", "\u043e", "\u043e\u0431", "\u043a", "\u043a\u0430\u043a", "\u044d\u0442\u043e", "\u044d\u0442\u043e\u0442",
        "the", "a", "an", "to", "in", "of", "and", "or", "for", "with",
    }
    tokens = [t for t in re.findall(r"[a-zA-Z\u0400-\u04FF0-9]+", lowered) if t and t not in stopwords]
    return " ".join(tokens)


def _characteristic_group_key(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ("character", "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436")):
        return "interests"
    if any(
        k in lowered
        for k in (
            "\u0430\u043d\u0438\u043c\u0435", "\u043c\u0430\u043d\u0433\u0430", "anime", "manga",
            "\u043c\u0443\u0437\u044b\u043a", "\u0436\u0430\u043d\u0440", "music", "genre", "\u0445\u043e\u0431\u0431\u0438", "hobby",
            "\u0440\u043e\u043a", "\u0440\u044d\u043f", "\u0445\u0438\u043f\u0445\u043e\u043f", "hiphop", "hip-hop", "rock", "rap", "\u0430\u043b\u044c\u0442\u0435\u0440\u043d\u0430\u0442\u0438\u0432",
        )
    ):
        return "interests"
    if any(k in lowered for k in ("\u0442\u043e\u043b\u043f", "\u0448\u0443\u043c", "\u0442\u0440\u0438\u0433\u0433\u0435\u0440", "\u0438\u0437\u0431\u0435\u0433", "\u0434\u0438\u0441\u043a\u043e\u043c\u0444\u043e\u0440\u0442", "\u0442\u0440\u0435\u0432\u043e\u0433", "\u0441\u043e\u043d", "\u0430\u043f\u043f\u0435\u0442\u0438\u0442", "crowd", "noise", "trigger", "avoid", "sleep", "appetite", "anxiety", "panic")):
        return "triggers"
    if any(k in lowered for k in ("\u0441\u043a\u0443\u0447", "\u043e\u0434\u0438\u043d\u043e\u043a", "\u0433\u0440\u0443\u0441\u0442", "\u044d\u043c\u043e\u0446", "mood", "emotion", "lonely", "misses")):
        return "emotional"
    if any(k in lowered for k in ("\u0441\u043f\u0440\u0430\u0432", "\u0440\u0435\u0441\u0443\u0440\u0441", "\u043f\u043e\u0434\u0434\u0435\u0440\u0436", "coping", "resource", "support")):
        return "resources"
    return "other"


def _characteristic_group_title(group_key: str, language: str) -> str:
    if language == "ru":
        labels = {
            "interests": "Интересы",
            "triggers": "Триггеры и симптомы",
            "emotional": "Эмоциональный контекст",
            "resources": "Ресурсы и опоры",
            "other": "Другое",
        }
    else:
        labels = {
            "interests": "Interests",
            "triggers": "Triggers & Symptoms",
            "emotional": "Emotional Context",
            "resources": "Resources",
            "other": "Other",
        }
    return labels.get(group_key, labels["other"])


def _group_characteristics(items: list[str], language: str = "en") -> list[dict]:
    order = ["interests", "triggers", "emotional", "resources", "other"]
    grouped: dict[str, list[str]] = {k: [] for k in order}
    seen_by_group: dict[str, dict[str, int]] = {k: {} for k in order}

    for raw in items:
        value = (raw or "").strip()
        if not value:
            continue
        group = _characteristic_group_key(value)
        key = _normalize_characteristic_key(value)
        if not key:
            key = value.lower()
        idx = seen_by_group[group].get(key)
        if idx is None:
            seen_by_group[group][key] = len(grouped[group])
            grouped[group].append(value)
            continue
        # Keep the more specific variant if duplicate by normalized meaning.
        if len(value) > len(grouped[group][idx]):
            grouped[group][idx] = value

    return [
        {
            "key": name,
            "title": _characteristic_group_title(name, language),
            "items": grouped[name],
        }
        for name in order
        if grouped[name]
    ]

templates.env.filters["markdown"] = lambda text: Markup(
    md.markdown(_cleanup_summary(text or ""), extensions=["extra", "sane_lists"])
)


@app.on_event('startup')
async def startup() -> None:
    ensure_tables()
    sync_patients_from_messages()
    if ADMIN_EMAIL and ADMIN_PASSWORD:
        existing = get_user_by_email(ADMIN_EMAIL)
        if not existing:
            create_user(ADMIN_EMAIL, hash_password(ADMIN_PASSWORD), role='admin')


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get('session')
    if not token:
        return None
    data = decode_session(token, SECRET_KEY)
    if not data:
        return None
    user = get_user_by_id(int(data.get('user_id', 0)))
    return user


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
    return user


def get_current_patient(request: Request) -> Optional[dict]:
    token = request.cookies.get('patient_session')
    if not token:
        return None
    data = decode_session(token, SECRET_KEY)
    if not data or data.get("scope") != "patient":
        return None
    chat_id = str(data.get("patient_chat_id", "")).strip()
    if not chat_id:
        return None
    return get_patient_by_chat_id(chat_id)


def require_patient(request: Request) -> dict:
    patient = get_current_patient(request)
    if not patient:
        raise HTTPException(status_code=401)
    return patient


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        if request.url.path.startswith('/patient'):
            return RedirectResponse(url='/patient/login', status_code=302)
        return RedirectResponse(url='/login', status_code=302)
    raise exc


@app.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse('login.html', {'request': request, 'error': None})


@app.post('/login')
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user['password_hash']):
        return templates.TemplateResponse('login.html', {'request': request, 'error': 'Invalid credentials'})
    token = encode_session({'user_id': user['id']}, SECRET_KEY, ttl_seconds=SESSION_TTL_SECONDS)
    response = RedirectResponse(url='/', status_code=302)
    response.set_cookie('session', token, httponly=True, samesite='lax')
    return response


@app.get('/logout')
async def logout():
    response = RedirectResponse(url='/login', status_code=302)
    response.delete_cookie('session')
    return response


@app.get('/patient/logout')
async def patient_logout():
    response = RedirectResponse(url='/patient/login', status_code=302)
    response.delete_cookie('patient_session')
    return response


@app.get('/patient/login', response_class=HTMLResponse)
async def patient_login_page(request: Request, code: str | None = None):
    return templates.TemplateResponse(
        'patient_login.html',
        {'request': request, 'error': None, 'code': code or ''},
    )


@app.post('/patient/login')
async def patient_login_submit(request: Request, code: str = Form(...)):
    chat_id = consume_patient_portal_token(code)
    if not chat_id:
        return templates.TemplateResponse(
            'patient_login.html',
            {'request': request, 'error': 'Invalid or expired code', 'code': code},
        )
    token = encode_session(
        {'scope': 'patient', 'patient_chat_id': chat_id},
        SECRET_KEY,
        ttl_seconds=PATIENT_SESSION_TTL_SECONDS,
    )
    response = RedirectResponse(url='/patient/preferences', status_code=302)
    response.set_cookie('patient_session', token, httponly=True, samesite='lax')
    return response


@app.get('/patient', response_class=HTMLResponse)
async def patient_home(patient: dict = Depends(require_patient)):
    return RedirectResponse(url='/patient/preferences', status_code=302)


@app.get('/patient/preferences', response_class=HTMLResponse)
async def patient_preferences_page(
    request: Request,
    patient: dict = Depends(require_patient),
    saved: str | None = None,
):
    language = get_patient_language(patient['chat_id'])
    preferences = get_patient_preferences(patient['chat_id'])
    pref_history = list_patient_preference_history(patient['chat_id'], limit=20)
    return templates.TemplateResponse(
        'patient_preferences.html',
        {
            'request': request,
            'patient': patient,
            'language': language,
            't': _patient_texts(language),
            'preferences': preferences,
            'pref_history': pref_history,
            'saved': saved == '1',
        },
    )


@app.post('/patient/preferences')
async def patient_preferences_update(
    request: Request,
    patient: dict = Depends(require_patient),
    tone: str = Form(...),
    response_length: str = Form(...),
    support_style: str = Form(...),
    proactive_frequency_hours: int = Form(...),
    quiet_hours_start: str = Form(""),
    quiet_hours_end: str = Form(""),
    topics_to_avoid: str = Form(""),
    proactive_enabled: str | None = Form(None),
):
    valid_tones = {"supportive", "calm", "direct"}
    valid_lengths = {"short", "medium", "detailed"}
    valid_styles = {"reflective", "practical", "motivational"}
    if tone not in valid_tones or response_length not in valid_lengths or support_style not in valid_styles:
        raise HTTPException(status_code=400)

    proactive_frequency_hours = max(6, min(168, int(proactive_frequency_hours)))
    hhmm = re.compile(r"^\d{2}:\d{2}$")
    q_start = quiet_hours_start.strip() or None
    q_end = quiet_hours_end.strip() or None
    if q_start and not hhmm.match(q_start):
        raise HTTPException(status_code=400)
    if q_end and not hhmm.match(q_end):
        raise HTTPException(status_code=400)

    upsert_patient_preferences(
        chat_id=patient['chat_id'],
        tone=tone,
        response_length=response_length,
        support_style=support_style,
        proactive_enabled=bool(proactive_enabled),
        proactive_frequency_hours=proactive_frequency_hours,
        quiet_hours_start=q_start,
        quiet_hours_end=q_end,
        topics_to_avoid=topics_to_avoid.strip(),
        actor_type="patient",
        actor_id=str(patient['chat_id']),
        source="patient_web",
    )
    return RedirectResponse(url='/patient/preferences?saved=1', status_code=302)


@app.get('/', response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: dict = Depends(require_user),
):
    patients = list_patients_for_user(user)
    active_alerts_list = list_alerts(active_only=True)
    active_alert_chat_ids = {a["chat_id"] for a in active_alerts_list}
    items = []
    for p in patients[:10]:
        last = get_last_summary(p['chat_id'])
        items.append({
            'patient': p,
            'last_summary': last,
        })
    hot_patients = []
    for p in patients:
        last = get_last_summary(p["chat_id"])
        if p["chat_id"] in active_alert_chat_ids or (last and last.get("risk_level") == "high"):
            hot_patients.append({"patient": p, "last_summary": last})
    return templates.TemplateResponse(
        'dashboard.html',
        {
            'request': request,
            'user': user,
            'debug_enabled': DEBUG_ENABLED,
            'patient_count': count_patients(),
            'active_alerts': count_active_alerts(),
            'recent_summaries': items,
            'hot_patients': hot_patients[:10],
        },
    )


@app.get('/patients', response_class=HTMLResponse)
async def patients_page(
    request: Request,
    user: dict = Depends(require_user),
    q: str | None = None,
    risk: str | None = None,
    emotion: str | None = None,
):
    patients = list_patients_for_user(user)
    items = []
    for p in patients:
        last = get_last_summary(p['chat_id'])
        if q:
            q_lower = q.lower()
            if q_lower not in p["display_name"].lower() and q_lower not in p["chat_id"].lower():
                continue
        if risk and last and last.get("risk_level") != risk:
            continue
        if emotion and last and last.get("emotion") != emotion:
            continue
        items.append({
            'patient': p,
            'last_summary': last,
        })
    return templates.TemplateResponse(
        'patients.html',
        {'request': request, 'user': user, 'items': items, 'q': q or '', 'risk': risk or '', 'emotion': emotion or ''},
    )


@app.get('/patients/{patient_id}', response_class=HTMLResponse)
async def patient_detail(request: Request, patient_id: int, user: dict = Depends(require_user)):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    summaries = list_summaries(patient['chat_id'])
    messages = list_messages(patient['chat_id'])
    patient_language = get_patient_language(patient['chat_id'])
    characteristics = list_patient_characteristics(patient['chat_id'])
    grouped_characteristics = _group_characteristics(characteristics, language=patient_language)
    preference_history = list_patient_preference_history(patient['chat_id'], limit=20)
    timeline = list_timeline(patient['chat_id'])
    return templates.TemplateResponse(
        'patient_detail.html',
        {
            'request': request,
            'user': user,
            'patient': patient,
            'summaries': summaries,
            'messages': messages,
            'patient_language': patient_language,
            'characteristics': characteristics,
            'grouped_characteristics': grouped_characteristics,
            'preference_history': preference_history,
            'debug_enabled': DEBUG_ENABLED,
            'timeline': timeline,
        },
    )


@app.get('/alerts', response_class=HTMLResponse)
async def alerts_page(request: Request, user: dict = Depends(require_user), show: str = "active", risk: str | None = None):
    active_only = show != "all"
    alerts = list_alerts(active_only=active_only)
    items = []
    for a in alerts:
        patient = get_patient_by_chat_id(a["chat_id"])
        last = get_last_summary(a["chat_id"])
        if risk and a["risk_level"] != risk:
            continue
        items.append(
            {
                "alert": a,
                "patient": patient,
                "last_summary": last,
            }
        )
    return templates.TemplateResponse(
        'alerts.html',
        {'request': request, 'user': user, 'items': items, 'show': show, 'risk': risk},
    )


@app.post('/alerts/{alert_id}/resolve')
async def alerts_resolve(alert_id: int, user: dict = Depends(require_user)):
    # Resolve and log under the patient's chat_id so it shows in History.
    from .db import get_alert_by_id
    alert = get_alert_by_id(alert_id)
    resolve_alert(alert_id)
    if alert:
        add_audit_event(
            chat_id=str(alert["chat_id"]),
            event_type="alert_resolved",
            message=f"Alert {alert_id} resolved by {user.get('email')}.",
        )
    return RedirectResponse(url='/alerts', status_code=302)


@app.get('/invites', response_class=HTMLResponse)
async def invites_page(request: Request, user: dict = Depends(require_user)):
    invites = list_invites(user['id'])
    return templates.TemplateResponse(
        'invites.html',
        {'request': request, 'user': user, 'invites': invites},
    )


@app.get('/settings', response_class=HTMLResponse)
async def settings_page(request: Request, user: dict = Depends(require_user)):
    report_language = get_user_report_language(user["id"])
    return templates.TemplateResponse(
        'settings.html',
        {'request': request, 'user': user, 'report_language': report_language},
    )


@app.post('/settings')
async def settings_update(request: Request, user: dict = Depends(require_user), report_language: str = Form(...)):
    if report_language not in ("en", "ru", "kk"):
        raise HTTPException(status_code=400)
    set_user_report_language(user["id"], report_language)
    return RedirectResponse(url='/settings', status_code=302)


@app.post('/invites')
async def invites_create(request: Request, user: dict = Depends(require_user)):
    create_invite(user['id'])
    return RedirectResponse(url='/invites', status_code=302)


@app.post('/invites/{code}/deactivate')
async def invites_deactivate(code: str, user: dict = Depends(require_user)):
    from .db import deactivate_invite
    deactivate_invite(code, user['id'])
    return RedirectResponse(url='/invites', status_code=302)


@app.post('/debug/patients/{patient_id}/clear-summaries')
async def debug_clear_patient_summaries(patient_id: int, user: dict = Depends(require_user)):
    if not DEBUG_ENABLED:
        raise HTTPException(status_code=404)
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403)
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    clear_summaries_for_chat(patient['chat_id'])
    return RedirectResponse(url=f'/patients/{patient_id}', status_code=302)


@app.post('/debug/patients/{patient_id}/clear-messages')
async def debug_clear_patient_messages(patient_id: int, user: dict = Depends(require_user)):
    if not DEBUG_ENABLED:
        raise HTTPException(status_code=404)
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403)
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    clear_messages_for_chat(patient['chat_id'])
    return RedirectResponse(url=f'/patients/{patient_id}', status_code=302)


@app.post('/debug/patients/{patient_id}/clear-timeline')
async def debug_clear_patient_timeline(patient_id: int, user: dict = Depends(require_user)):
    if not DEBUG_ENABLED:
        raise HTTPException(status_code=404)
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403)
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    clear_timeline_for_chat(patient['chat_id'])
    return RedirectResponse(url=f'/patients/{patient_id}', status_code=302)


# API
@app.get('/api/patients')
async def api_patients(user: dict = Depends(require_user)):
    return JSONResponse(list_patients_for_user(user))


@app.get('/api/patients/{patient_id}/summaries')
async def api_summaries(patient_id: int, user: dict = Depends(require_user)):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    return JSONResponse(list_summaries(patient['chat_id']))


@app.get('/api/patients/{patient_id}/messages')
async def api_messages(patient_id: int, user: dict = Depends(require_user)):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    return JSONResponse(list_messages(patient['chat_id']))


@app.get('/api/alerts')
async def api_alerts(user: dict = Depends(require_user)):
    return JSONResponse(list_alerts(active_only=True))


@app.get('/export/alerts.csv')
async def export_alerts_csv(user: dict = Depends(require_user)):
    alerts = list_alerts(active_only=False)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "chat_id", "reason", "risk_level", "created_at", "resolved_at"])
    for a in alerts:
        writer.writerow([a.get("id"), a.get("chat_id"), a.get("reason"), a.get("risk_level"), a.get("created_at"), a.get("resolved_at")])
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=alerts.csv"})


@app.get('/patients/{patient_id}/export.csv')
async def export_patient_csv(patient_id: int, user: dict = Depends(require_user)):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404)
    summaries = list_summaries(patient["chat_id"])
    messages = list_messages(patient["chat_id"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["type", "role", "text", "emotion", "risk_level", "created_at"])
    for m in messages:
        writer.writerow(["message", m.get("role"), m.get("text"), "", "", m.get("created_at")])
    for s in summaries:
        writer.writerow(["summary", "", s.get("summary"), s.get("emotion"), s.get("risk_level"), s.get("created_at")])
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=patient_{patient_id}.csv"})
