# Mental Health AI

## Structure
- `telegram_bot/` - Telegram bot (dialog, analysis, reports).
- `healthcare_web/` - Specialist web dashboard (FastAPI + Jinja).
- `data/dialogs.db` - Shared SQLite DB (messages, summaries, alerts, invites, history).

## 1) Run Telegram bot

### Install deps
```
pip install -r telegram_bot/requirements.txt
```

### Env vars
Create `telegram_bot/.env`:
```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
MODEL=gpt-4o-mini
PATIENT_PORTAL_BASE_URL=http://127.0.0.1:8000
PATIENT_PORTAL_TOKEN_TTL_MINUTES=30
```

### Start
```
python telegram_bot/main.py
```

### Language
- `/start` - choose language (en/ru/kk)
- `/language` - change language later

Patient language affects bot replies. Report language is configured per doctor in the web dashboard (Settings).

---

## 2) Run web dashboard

### Install deps
```
pip install -r healthcare_web/requirements.txt
```

### Env vars
Create `healthcare_web/.env`:
```
WEB_SECRET_KEY=change-me
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-me
WEB_SESSION_TTL_SECONDS=86400
PATIENT_WEB_SESSION_TTL_SECONDS=604800
DEBUG=True
```

### Start
From `healthcare_web` folder:
```
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

---

## 3) Patient -> doctor link (invite flow)

1. In web, open `/invites` and create invite.
2. Patient in bot: `/link` -> enters code.
3. Patient appears in doctor list.

Invite becomes inactive after use; status is visible on `/invites`.

---

## 4) Patient personalization web

1. In Telegram bot, patient sends `/portal`.
2. Bot returns one-time login code + link to `/patient/login`.
3. Patient opens `/patient/preferences` and configures tone, response length, support style, proactive frequency, quiet hours, and topics to avoid.
4. Bot uses these preferences in normal replies and proactive check-ins.

---

## Notes
- History shows only new events after logging is enabled.
- Debug clear buttons appear only when `DEBUG=True` and only on patient page.
