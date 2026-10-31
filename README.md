# Mental Health AI

## Structure
- `telegram_bot/` - Telegram bot (dialog, analysis, reports).
- `healthcare_web/` - Specialist web dashboard (FastAPI + Jinja).
- `data/dialogs.db` - Shared SQLite DB (messages, summaries, alerts, invites, history). Created automatically, not stored in git.
- `scripts/run_baselines.py` - keyword / single-prompt LLM baselines for the evaluation datasets.
- `tests/` - pytest suite (no network or API keys needed).

## 1) Run Telegram bot

### Install deps
```
pip install -r telegram_bot/requirements.txt
```
The `job-queue` extra of `python-telegram-bot` is required for proactive check-ins; without it the bot logs a warning and skips them.

### Env vars
Copy `telegram_bot/.env.example` to `telegram_bot/.env` and fill in at least:
```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
MODEL=gpt-6.1-sol
REASONING_EFFORT=low
PATIENT_PORTAL_BASE_URL=http://127.0.0.1:8000
PATIENT_PORTAL_TOKEN_TTL_MINUTES=30
CRISIS_PHONE=112
```
The file is found regardless of the current directory; a project-root `.env` works as a fallback.

### Start
```
python telegram_bot/main.py
```

### Commands
- `/start` - choose language (en/ru/kk) and register
- `/language` - change language later
- `/link` - link to a specialist with an invite code
- `/portal` - get a one-time login code for the patient settings portal

Patient language affects bot replies. Report language is configured per doctor in the web dashboard (Settings).

### Risk handling
- Every message is analysed. Explicit self-harm / suicidal statements (en/ru/kk) force `high` risk even if the model under-rates them or the model call fails.
- A `high` risk message immediately shows the crisis options and creates an alert for the specialist (one active alert per patient; new ones are created after the previous is resolved).
- Every 3 user messages a summary is generated for the specialist; it can also raise an alert.

---

## 2) Run web dashboard

### Install deps
```
pip install -r healthcare_web/requirements.txt
```

### Env vars
Create `healthcare_web/.env`:
```
WEB_SECRET_KEY=<long random string>
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong password>
WEB_SESSION_TTL_SECONDS=86400
PATIENT_WEB_SESSION_TTL_SECONDS=604800
COOKIE_SECURE=false   # set to true behind HTTPS
DEBUG=False
```
Generate a key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. If `WEB_SECRET_KEY` is missing, a random key is used and sessions reset on every restart.

### Start
From `healthcare_web` folder:
```
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

### Access model
- `admin` sees all patients and alerts.
- A specialist sees only patients linked to them through an invite; other patients' pages, API, CSV exports and alerts return 404.

---

## 3) Patient -> doctor link (invite flow)

1. In web, open `/invites` and create invite.
2. Patient in bot: `/link` -> enters code (case-insensitive).
3. Patient appears in doctor list.

Invite becomes inactive after use; status is visible on `/invites`.

---

## 4) Patient personalization web

1. In Telegram bot, patient sends `/portal`.
2. Bot returns one-time login code + link to `/patient/login`.
3. Patient opens `/patient/preferences` and configures tone, response length, support style, proactive frequency, quiet hours, and topics to avoid.
4. Bot uses these preferences in normal replies and proactive check-ins.

Quiet hours are interpreted in the server's local time.

---

## 5) Tests
```
pip install -r requirements-dev.txt
pytest
```

---

## Notes
- History shows only new events after logging is enabled.
- Debug clear buttons appear only when `DEBUG=True` and only on patient page (admin only).
- Logs at `INFO` level do not contain patient message content; `DEBUG` does.
