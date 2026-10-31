import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.settings import settings
from src.core.emotion_schema import EmotionResult
from src.core.graph import (
    generate_proactive_checkin,
    run_graph_with_analysis,
    should_create_alert,
    summarize_dialog,
)
from src.storage.dialog_store import DialogStore


SUPPORTED_LANGUAGES = ("en", "ru", "kk")

REGISTRATION_STATE_KEYS = (
    "awaiting_consent",
    "awaiting_full_name",
    "awaiting_age",
    "awaiting_gender",
    "awaiting_phone",
    "awaiting_trusted",
    "awaiting_notes",
    "awaiting_link_code",
    "profile_full_name",
    "profile_age",
    "profile_gender",
    "profile_phone",
    "profile_trusted",
)


def _utcnow() -> datetime:
    # SQLite CURRENT_TIMESTAMP is UTC; compare against naive UTC, not local time.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TelegramBot:
    """Wrapper over python-telegram-bot for running the dialog agent."""

    MIN_MESSAGES_FOR_REPORT = 3
    PHONE_MIN_LEN = 7
    PHONE_MAX_LEN = 15

    def __init__(self, token: str):
        self.store = DialogStore()
        self.app = (
            ApplicationBuilder()
            .token(token)
            .post_init(self.on_startup)
            .post_shutdown(self.on_shutdown)
            .build()
        )
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("language", self.language_start))
        self.app.add_handler(CommandHandler("link", self.link_start))
        self.app.add_handler(CommandHandler("portal", self.portal_start))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def on_startup(self, app: Application) -> None:
        """Log bot startup."""
        logging.info("Bot is starting...")
        if not settings.proactive_enabled:
            return
        if not app.job_queue:
            logging.warning(
                "Proactive outreach is enabled but JobQueue is unavailable. "
                "Install python-telegram-bot[job-queue] to enable it."
            )
            return
        app.job_queue.run_repeating(
            self._run_proactive_outreach,
            interval=max(60, settings.proactive_check_interval_minutes * 60),
            first=30,
            name="proactive_outreach",
        )
        logging.info(
            "Scheduled proactive outreach job: every %s min, idle threshold=%s h",
            settings.proactive_check_interval_minutes,
            settings.proactive_idle_hours,
        )

    async def on_shutdown(self, app: Application) -> None:
        """Log bot shutdown."""
        logging.info("Bot is stopping.")

    async def _run_proactive_outreach(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Pre-filter with the smallest possible interval; the per-chat smart
        # interval (>= 6 h) is applied below.
        candidates = self.store.list_inactive_chats(
            idle_hours=1,
            min_user_messages=settings.proactive_min_user_messages,
            limit=settings.proactive_batch_size,
        )
        if not candidates:
            return

        for candidate in candidates:
            chat_id_str = str(candidate.get("chat_id", "")).strip()
            if not chat_id_str:
                continue
            try:
                chat_id = int(chat_id_str)
            except ValueError:
                continue

            language = candidate.get("language") or self.store.get_language(chat_id)
            preferences = self.store.get_patient_preferences(chat_id)
            if not bool(preferences.get("proactive_enabled", True)):
                continue
            if self._is_quiet_hours_now(
                preferences.get("quiet_hours_start"),
                preferences.get("quiet_hours_end"),
            ):
                continue
            base_hours = int(preferences.get("proactive_frequency_hours") or settings.proactive_idle_hours)
            summaries = self.store.get_recent_summaries(chat_id, limit=3)
            smart_hours = self._compute_smart_interval_hours(base_hours, summaries)
            last_message_at = self._parse_db_datetime(str(candidate.get("last_message_at") or ""))
            last_outreach_at = self._parse_db_datetime(str(candidate.get("last_outreach_at") or ""))
            reference_dt = last_message_at
            if last_outreach_at and (reference_dt is None or last_outreach_at > reference_dt):
                reference_dt = last_outreach_at
            if not reference_dt:
                continue
            if (_utcnow() - reference_dt).total_seconds() < smart_hours * 3600:
                continue
            recent = self.store.get_recent_messages(chat_id, limit=8)
            history = [f"{m['role']}: {m['text']}" for m in recent]
            text = await generate_proactive_checkin(
                history=history,
                language=language,
                preferences=preferences,
            )
            try:
                await context.bot.send_message(chat_id=chat_id, text=text)
                self.store.add_message(chat_id, "assistant", text)
                self.store.mark_proactive_outreach(chat_id)
                self.store.add_event(
                    chat_id=chat_id_str,
                    event_type="proactive_checkin_sent",
                    message=f"Proactive check-in sent (smart_interval={smart_hours}h, base={base_hours}h).",
                )
            except Exception:
                logging.exception("Failed to send proactive check-in to chat_id=%s", chat_id_str)
                # Still mark the attempt, otherwise a blocked bot is retried every cycle.
                self.store.mark_proactive_outreach(chat_id)

    def _get_language(self, chat_id: int) -> str:
        language = self.store.get_language(chat_id)
        return language if language in SUPPORTED_LANGUAGES else "en"

    def _has_specialist(self, chat_id: int) -> bool:
        return self.store.has_specialist(chat_id)

    @staticmethod
    def _localized(texts: dict, language: str) -> str:
        return texts.get(language, texts["en"])

    def _consent_text(self, language: str) -> str:
        texts = {
            "ru": (
                "Перед регистрацией нужно согласие на обработку данных. "
                "Вы подтверждаете, что предоставляете свои контактные данные и контакт доверенного человека, "
                "и разрешаете специалисту видеть эту информацию в платформе?\n\n"
                "Ответьте: да / нет"
            ),
            "en": (
                "Before registration, we need your consent to process data. "
                "You confirm you provide your contact info and a trusted contact, "
                "and allow a specialist to view this information on the platform?\n\n"
                "Reply: yes / no"
            ),
            "kk": (
                "Тіркелу алдында деректерді өңдеуге келісіміңіз қажет. "
                "Сіз өз байланыс деректеріңізді және сенімді адамның байланысын беретініңізді, "
                "әрі маманға бұл ақпаратты платформада көруге рұқсат беретініңізді растайсыз ба?\n\n"
                "Жауап беріңіз: иә / жоқ"
            ),
        }
        return self._localized(texts, language)

    def _prompt_text(self, key: str, language: str) -> str:
        prompts = {
            "full_name": {
                "ru": "Введите ваше полное имя.",
                "en": "Enter your full name.",
                "kk": "Толық аты-жөніңізді енгізіңіз.",
            },
            "age": {
                "ru": "Введите ваш возраст (число).",
                "en": "Enter your age (number).",
                "kk": "Жасыңызды енгізіңіз (сан).",
            },
            "gender": {
                "ru": "Выберите пол кнопкой ниже.",
                "en": "Choose gender using the buttons below.",
                "kk": "Төмендегі батырма арқылы жынысыңызды таңдаңыз.",
            },
            "phone": {
                "ru": "Введите ваш номер телефона (обязателен).",
                "en": "Enter your phone number (required).",
                "kk": "Телефон нөміріңізді енгізіңіз (міндетті).",
            },
            "trusted": {
                "ru": "Введите контакт доверенного человека (телефон/имя).",
                "en": "Enter a trusted contact (phone/name).",
                "kk": "Сенімді адамның байланысын енгізіңіз (телефон/аты).",
            },
            "notes": {
                "ru": "Дополнительные примечания (необязательно). Напишите '-' чтобы пропустить.",
                "en": "Additional notes (optional). Type '-' to skip.",
                "kk": "Қосымша ескертпелер (міндетті емес). Өткізіп жіберу үшін '-' деп жазыңыз.",
            },
            "registered": {
                "ru": "Регистрация завершена. Спасибо!",
                "en": "Registration complete. Thank you!",
                "kk": "Тіркелу аяқталды. Рақмет!",
            },
            "invalid_age": {
                "ru": "Возраст должен быть числом от 5 до 120. Попробуйте снова.",
                "en": "Age must be a number between 5 and 120. Try again.",
                "kk": "Жас 5 пен 120 аралығындағы сан болуы керек. Қайталап көріңіз.",
            },
            "invalid_gender": {
                "ru": "Пожалуйста выберите: мужской / женский / другое.",
                "en": "Please choose: male / female / other.",
                "kk": "Таңдаңыз: ер / әйел / басқа.",
            },
            "invalid_phone": {
                "ru": "Номер телефона некорректен. Введите в формате +XXXXXXXXXXX.",
                "en": "Phone number is invalid. Enter in +XXXXXXXXXXX format.",
                "kk": "Телефон нөмірі қате. +XXXXXXXXXXX форматында енгізіңіз.",
            },
            "consent_required": {
                "ru": (
                    "Без согласия регистрация невозможна. Вы можете продолжить общение без регистрации "
                    "и зарегистрироваться позже командой /start."
                ),
                "en": (
                    "Consent is required to register. You can keep chatting without registration "
                    "and register later with /start."
                ),
                "kk": (
                    "Келісімсіз тіркелу мүмкін емес. Тіркелмей-ақ сөйлесуді жалғастыра аласыз, "
                    "кейін /start командасымен тіркеле аласыз."
                ),
            },
            "link_prompt": {
                "ru": "Введите код приглашения от специалиста, чтобы привязать аккаунт.",
                "en": "Please enter the doctor invite code to link your account.",
                "kk": "Аккаунтты байланыстыру үшін маманның шақыру кодын енгізіңіз.",
            },
            "link_linked": {
                "ru": "Аккаунт успешно привязан к специалисту.",
                "en": "Linked successfully.",
                "kk": "Аккаунт маманға сәтті байланыстырылды.",
            },
            "link_invalid": {
                "ru": "Неверный код приглашения.",
                "en": "Invalid invite code.",
                "kk": "Шақыру коды қате.",
            },
            "link_used": {
                "ru": "Этот код приглашения уже использован.",
                "en": "This invite code is already used.",
                "kk": "Бұл шақыру коды бұрын пайдаланылған.",
            },
            "link_expired": {
                "ru": "Срок действия кода приглашения истёк.",
                "en": "This invite code has expired.",
                "kk": "Шақыру кодының мерзімі өтіп кеткен.",
            },
            "language_set": {
                "ru": "Язык: русский",
                "en": "Language: English",
                "kk": "Тіл: қазақша",
            },
            "error": {
                "ru": "Извините, что-то пошло не так. Попробуйте ещё раз.",
                "en": "Sorry, something went wrong. Please try again.",
                "kk": "Кешіріңіз, қате орын алды. Қайталап көріңіз.",
            },
        }
        texts = prompts.get(key, {})
        return texts.get(language, texts.get("en", ""))

    def _language_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("English", callback_data="lang:en"),
                    InlineKeyboardButton("Русский", callback_data="lang:ru"),
                    InlineKeyboardButton("Қазақша", callback_data="lang:kk"),
                ]
            ]
        )

    def _gender_keyboard(self, language: str) -> InlineKeyboardMarkup:
        labels = {
            "en": ("Male", "Female", "Other"),
            "ru": ("Мужской", "Женский", "Другое"),
            "kk": ("Ер", "Әйел", "Басқа"),
        }
        male, female, other = self._localized(labels, language)
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(male, callback_data="gender:male"),
                    InlineKeyboardButton(female, callback_data="gender:female"),
                    InlineKeyboardButton(other, callback_data="gender:other"),
                ]
            ]
        )

    def _normalize_phone(self, text: str) -> str:
        phone = text.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if phone.startswith("00"):
            phone = "+" + phone[2:]
        return phone

    def _is_valid_phone(self, phone: str) -> bool:
        if phone.startswith("+"):
            digits = phone[1:]
        else:
            digits = phone
        if not digits.isdigit():
            return False
        return self.PHONE_MIN_LEN <= len(digits) <= self.PHONE_MAX_LEN

    def _parse_gender(self, text: str) -> str | None:
        value = text.strip().lower()
        if value in ("male", "m", "man", "м", "муж", "мужской", "ер"):
            return "male"
        if value in ("female", "f", "woman", "ж", "жен", "женский", "әйел"):
            return "female"
        if value in ("other", "o", "другое", "другой", "иной", "басқа"):
            return "other"
        return None

    def _portal_text(self, language: str, link: str, token: str, ttl_minutes: int) -> str:
        if language == "ru":
            return (
                "Откройте портал персональных настроек:\n"
                f"{link}\n\n"
                f"Код для входа: {token}\n"
                f"Код действует {ttl_minutes} мин."
            )
        if language == "kk":
            return (
                "Жеке баптаулар порталын ашыңыз:\n"
                f"{link}\n\n"
                f"Кіру коды: {token}\n"
                f"Код {ttl_minutes} минут жарамды."
            )
        return (
            "Open your personal settings portal:\n"
            f"{link}\n\n"
            f"Login code: {token}\n"
            f"Code expires in {ttl_minutes} minutes."
        )

    def _is_quiet_hours_now(self, start: str | None, end: str | None) -> bool:
        if not start or not end:
            return False
        if not re.match(r"^\d{2}:\d{2}$", start) or not re.match(r"^\d{2}:\d{2}$", end):
            return False
        # Quiet hours are entered as wall-clock time, so compare with server local time.
        now = datetime.now().time()
        try:
            start_t = datetime.strptime(start, "%H:%M").time()
            end_t = datetime.strptime(end, "%H:%M").time()
        except ValueError:
            return False
        if start_t == end_t:
            return False
        if start_t < end_t:
            return start_t <= now < end_t
        # Overnight range (e.g., 22:00 -> 08:00)
        return now >= start_t or now < end_t

    def _parse_db_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _summary_severity_score(self, summary: dict) -> float:
        risk_map = {"low": 0.2, "medium": 0.55, "medium_high": 0.7, "high": 0.9}
        risk = risk_map.get(str(summary.get("risk_level", "low")).lower(), 0.2)
        dep = float(summary.get("depression_score") or 0.0)
        anx = float(summary.get("anxiety_score") or 0.0)
        stress = float(summary.get("stress_score") or 0.0)
        metric = max(dep, anx, stress)
        return max(risk, metric)

    def _compute_smart_interval_hours(self, base_hours: int, summaries: List[dict]) -> int:
        base = max(6, min(168, int(base_hours)))
        if not summaries:
            return base
        latest = summaries[0]
        latest_risk = str(latest.get("risk_level", "low")).lower()
        risk_multiplier = {"high": 0.6, "medium_high": 0.75, "medium": 0.85, "low": 1.1}.get(latest_risk, 1.0)
        trend_multiplier = 1.0
        if len(summaries) >= 2:
            current_score = self._summary_severity_score(summaries[0])
            prev_score = self._summary_severity_score(summaries[1])
            delta = current_score - prev_score
            if delta > 0.12:
                trend_multiplier *= 0.8
            elif delta < -0.12:
                trend_multiplier *= 1.25
            prev_risk = str(summaries[1].get("risk_level", "low")).lower()
            risk_rank = {"low": 1, "medium": 2, "medium_high": 3, "high": 4}
            if risk_rank.get(latest_risk, 1) > risk_rank.get(prev_risk, 1):
                trend_multiplier *= 0.85
        smart = int(round(base * risk_multiplier * trend_multiplier))
        return max(6, min(168, smart))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Restarting must not leave the user stuck in a half-finished registration step.
        for key in REGISTRATION_STATE_KEYS:
            context.user_data.pop(key, None)
        await self.language_start(update, context)

    async def language_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.user_data["awaiting_language"] = True
        await update.message.reply_text(
            "Choose language / Выберите язык / Тілді таңдаңыз:",
            reply_markup=self._language_keyboard(),
        )

    async def link_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        context.user_data["awaiting_link_code"] = True
        await update.message.reply_text(
            self._prompt_text("link_prompt", self._get_language(chat_id))
        )

    async def portal_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not chat_id:
            return
        language = self._get_language(chat_id)
        ttl_minutes = int(settings.patient_portal_token_ttl_minutes)
        token = self.store.create_patient_portal_token(
            chat_id=chat_id,
            ttl_minutes=ttl_minutes,
        )
        base = settings.patient_portal_base_url.rstrip("/")
        link = f"{base}/patient/login?code={token}"
        await update.message.reply_text(self._portal_text(language, link, token, ttl_minutes))

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        await query.answer()
        chat_id = query.message.chat.id if query.message else 0
        data = query.data

        if data.startswith("lang:") and context.user_data.get("awaiting_language"):
            lang = data.split(":", 1)[1]
            if lang not in SUPPORTED_LANGUAGES:
                return
            self.store.set_language(chat_id, lang)
            context.user_data["awaiting_language"] = False
            await query.edit_message_text(self._prompt_text("language_set", lang))
            profile = self.store.get_patient_profile(chat_id)
            if profile and profile.get("phone") and profile.get("consent_at"):
                return
            context.user_data["awaiting_consent"] = True
            await query.message.reply_text(self._consent_text(lang))
            return

        if data.startswith("gender:") and context.user_data.get("awaiting_gender"):
            language = self._get_language(chat_id)
            gender = data.split(":", 1)[1]
            if gender not in ("male", "female", "other"):
                await query.message.reply_text(self._prompt_text("invalid_gender", language))
                return
            self._accept_gender(context, gender)
            await query.edit_message_text(self._prompt_text("gender", language))
            await query.message.reply_text(self._prompt_text("phone", language))
            return

        if data.startswith("crisis:"):
            language = self._get_language(chat_id)
            action = data.split(":", 1)[1]
            phone = settings.crisis_phone
            if action == "call":
                await query.message.reply_text(
                    self._localized(
                        {
                            "en": f"Call {phone} (emergency services).",
                            "ru": f"Позвоните {phone} (экстренные службы).",
                            "kk": f"{phone} нөміріне қоңырау шалыңыз (жедел қызметтер).",
                        },
                        language,
                    )
                )
                return
            if action == "safe":
                await query.message.reply_text(
                    self._localized(
                        {
                            "en": "Okay. Thank you for letting me know.",
                            "ru": "Хорошо. Спасибо, что сообщили.",
                            "kk": "Жақсы. Хабарлағаныңызға рақмет.",
                        },
                        language,
                    )
                )
                self.store.add_event(
                    chat_id=str(chat_id),
                    event_type="crisis_marked_safe",
                    message="Patient pressed 'I'm safe' after a crisis prompt.",
                )
                return
            if action == "trusted":
                profile = self.store.get_patient_profile(chat_id)
                contact = profile.get("trusted_contact") if profile else None
                if contact:
                    text = self._localized(
                        {
                            "en": f"Trusted contact: {contact}",
                            "ru": f"Контакт доверенного: {contact}",
                            "kk": f"Сенімді адамның байланысы: {contact}",
                        },
                        language,
                    )
                else:
                    text = self._localized(
                        {
                            "en": "Trusted contact not found. You can add it during registration (/start).",
                            "ru": "Доверенный контакт не найден. Вы можете добавить его через регистрацию (/start).",
                            "kk": "Сенімді байланыс табылмады. Оны тіркелу кезінде қосуға болады (/start).",
                        },
                        language,
                    )
                await query.message.reply_text(text)
                return
            if action == "specialist":
                if self._has_specialist(chat_id):
                    text = self._localized(
                        {
                            "en": "Your specialist has received an alert. If you have their contact, you can also reach out directly now.",
                            "ru": "Ваш специалист получил сигнал. Если у вас есть его контакт, можете написать ему прямо сейчас.",
                            "kk": "Маманыңызға белгі жіберілді. Егер оның байланысы болса, оған қазір тікелей жаза аласыз.",
                        },
                        language,
                    )
                else:
                    text = self._localized(
                        {
                            "en": "If you have your specialist's contact, please reach out now. You can link a specialist with /link.",
                            "ru": "Если у вас есть контакт специалиста, пожалуйста, свяжитесь с ним сейчас. Привязать специалиста можно командой /link.",
                            "kk": "Егер маманның байланысы болса, оған қазір хабарласыңыз. Маманды /link командасымен байланыстыруға болады.",
                        },
                        language,
                    )
                await query.message.reply_text(text)
                return

    def _accept_gender(self, context: ContextTypes.DEFAULT_TYPE, gender: str) -> None:
        context.user_data["profile_gender"] = gender
        context.user_data["awaiting_gender"] = False
        context.user_data["awaiting_phone"] = True

    def _crisis_text(self, language: str, has_specialist: bool) -> str:
        texts = {
            "ru": (
                "Похоже, сейчас очень тяжело. Я рядом.\n"
                "Если есть риск для вашей безопасности, пожалуйста, обратитесь за срочной помощью.\n\n"
                "Вы можете выбрать один из вариантов ниже."
            ),
            "en": (
                "It sounds really hard right now. I'm here.\n"
                "If you feel unsafe, please seek urgent help.\n\n"
                "You can choose one of the options below."
            ),
            "kk": (
                "Қазір сізге өте ауыр сияқты. Мен осындамын.\n"
                "Егер қауіпсіздігіңізге қатер болса, шұғыл көмекке жүгініңіз.\n\n"
                "Төмендегі нұсқалардың бірін таңдай аласыз."
            ),
        }
        base = self._localized(texts, language)
        if has_specialist:
            extra = {
                "ru": "\n\nВаш специалист увидит этот сигнал; вы также можете написать ему напрямую.",
                "en": "\n\nYour specialist will see this signal; you can also message them directly.",
                "kk": "\n\nМаманыңыз бұл белгіні көреді; оған тікелей жаза да аласыз.",
            }
            return base + self._localized(extra, language)
        return base

    def _crisis_keyboard(self, language: str) -> InlineKeyboardMarkup:
        phone = settings.crisis_phone
        labels = {
            "ru": (f"Позвонить {phone}", "Написать доверенному", "Связаться со специалистом", "Я в безопасности"),
            "en": ("Call emergency", "Message trusted contact", "Contact specialist", "I'm safe"),
            "kk": (f"{phone} нөміріне қоңырау", "Сенімді адамға жазу", "Маманмен байланысу", "Мен қауіпсіздемін"),
        }
        call, trusted, specialist, safe = self._localized(labels, language)
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(call, callback_data="crisis:call")],
                [InlineKeyboardButton(trusted, callback_data="crisis:trusted")],
                [InlineKeyboardButton(specialist, callback_data="crisis:specialist")],
                [InlineKeyboardButton(safe, callback_data="crisis:safe")],
            ]
        )

    def _create_alert_once(self, chat_id: int, reason: str, analysis: EmotionResult) -> bool:
        """Creates an alert unless the patient already has an unresolved one."""
        if self.store.has_active_alert(chat_id):
            return False
        self.store.add_alert(
            chat_id=chat_id,
            reason=reason,
            risk_level=analysis.risk_level,
            emotion=analysis.emotion,
        )
        return True

    async def _send_crisis_prompt(self, update: Update, chat_id: int, language: str) -> None:
        await update.message.reply_text(
            self._crisis_text(language, self._has_specialist(chat_id)),
            reply_markup=self._crisis_keyboard(language),
        )

    async def process_message(self, chat_id: int, text: str, history: Optional[List[str]] = None) -> Optional[str]:
        language = self._get_language(chat_id)
        preferences = self.store.get_patient_preferences(chat_id)
        response, _ = await run_graph_with_analysis(
            text,
            history=history or [],
            language=language,
            has_specialist=self._has_specialist(chat_id),
            preferences=preferences,
        )
        return response

    @staticmethod
    def _display_name(update: Update) -> str:
        if update.effective_user:
            return (
                update.effective_user.full_name
                or update.effective_user.username
                or "Patient"
            )
        return "Patient"

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        try:
            text = update.message.text

            if context.user_data.get("awaiting_language"):
                await update.message.reply_text(
                    "Please use the language buttons / Выберите язык кнопками / Тілді батырмамен таңдаңыз.",
                    reply_markup=self._language_keyboard(),
                )
                return

            if context.user_data.get("awaiting_consent"):
                language = self._get_language(chat_id)
                value = text.strip().lower()
                context.user_data["awaiting_consent"] = False
                if value in ("yes", "y", "да", "согласен", "согласна", "иә", "ия", "келісемін"):
                    context.user_data["awaiting_full_name"] = True
                    await update.message.reply_text(self._prompt_text("full_name", language))
                else:
                    await update.message.reply_text(self._prompt_text("consent_required", language))
                return

            if context.user_data.get("awaiting_full_name"):
                language = self._get_language(chat_id)
                context.user_data["profile_full_name"] = text.strip()
                context.user_data["awaiting_full_name"] = False
                context.user_data["awaiting_age"] = True
                await update.message.reply_text(self._prompt_text("age", language))
                return

            if context.user_data.get("awaiting_age"):
                language = self._get_language(chat_id)
                try:
                    age = int(text.strip())
                except ValueError:
                    await update.message.reply_text(self._prompt_text("invalid_age", language))
                    return
                if age < 5 or age > 120:
                    await update.message.reply_text(self._prompt_text("invalid_age", language))
                    return
                context.user_data["profile_age"] = age
                context.user_data["awaiting_age"] = False
                context.user_data["awaiting_gender"] = True
                await update.message.reply_text(
                    self._prompt_text("gender", language),
                    reply_markup=self._gender_keyboard(language),
                )
                return

            if context.user_data.get("awaiting_gender"):
                language = self._get_language(chat_id)
                gender = self._parse_gender(text)
                if gender:
                    self._accept_gender(context, gender)
                    await update.message.reply_text(self._prompt_text("phone", language))
                    return
                await update.message.reply_text(
                    self._prompt_text("gender", language),
                    reply_markup=self._gender_keyboard(language),
                )
                return

            if context.user_data.get("awaiting_phone"):
                language = self._get_language(chat_id)
                phone = self._normalize_phone(text)
                if not self._is_valid_phone(phone):
                    await update.message.reply_text(self._prompt_text("invalid_phone", language))
                    return
                context.user_data["profile_phone"] = phone
                context.user_data["awaiting_phone"] = False
                context.user_data["awaiting_trusted"] = True
                await update.message.reply_text(self._prompt_text("trusted", language))
                return

            if context.user_data.get("awaiting_trusted"):
                language = self._get_language(chat_id)
                context.user_data["profile_trusted"] = text.strip()
                context.user_data["awaiting_trusted"] = False
                context.user_data["awaiting_notes"] = True
                await update.message.reply_text(self._prompt_text("notes", language))
                return

            if context.user_data.get("awaiting_notes"):
                language = self._get_language(chat_id)
                notes = text.strip()
                if notes.lower() in ("-", "skip", "нет", "no", "жоқ"):
                    notes = None
                display_name = self._display_name(update)
                self.store.upsert_patient_profile(
                    chat_id=chat_id,
                    display_name=display_name,
                    full_name=context.user_data.get("profile_full_name", display_name),
                    age=int(context.user_data.get("profile_age", 0)),
                    gender=context.user_data.get("profile_gender", "other"),
                    phone=context.user_data.get("profile_phone", ""),
                    trusted_contact=context.user_data.get("profile_trusted", ""),
                    notes=notes,
                )
                for key in REGISTRATION_STATE_KEYS:
                    context.user_data.pop(key, None)
                await update.message.reply_text(self._prompt_text("registered", language))
                return

            if context.user_data.get("awaiting_link_code"):
                context.user_data["awaiting_link_code"] = False
                language = self._get_language(chat_id)
                username = None
                if update.effective_user:
                    username = update.effective_user.username or update.effective_user.full_name
                ok, status = self.store.link_patient_with_invite(
                    code=text,
                    chat_id=chat_id,
                    display_name=self._display_name(update),
                    username=username,
                )
                await update.message.reply_text(self._prompt_text(f"link_{status}", language))
                return

            self.store.add_message(chat_id, "user", text)

            # Use recent messages for context.
            recent = self.store.get_recent_messages(chat_id, limit=12)
            if recent:
                current_text = recent[-1]["text"]
                history = [
                    f"{m['role']}: {m['text']}"
                    for m in recent[:-1]
                ]
            else:
                current_text = text
                history = []

            # Main assistant reply.
            language = self._get_language(chat_id)
            preferences = self.store.get_patient_preferences(chat_id)
            result, message_analysis = await run_graph_with_analysis(
                current_text,
                history=history,
                language=language,
                has_specialist=self._has_specialist(chat_id),
                preferences=preferences,
            )
            self.store.add_message(chat_id, "assistant", result)
            await update.message.reply_text(result)

            # Acute risk in a single message must not wait for the periodic summary.
            crisis_prompt_sent = False
            if str(message_analysis.risk_level).strip().lower() == "high":
                self._create_alert_once(chat_id, "high_risk_message", message_analysis)
                await self._send_crisis_prompt(update, chat_id, language)
                crisis_prompt_sent = True

            # Summary threshold for specialists (user does not receive it).
            user_count = self.store.get_user_message_count(chat_id)
            last_report_count = self.store.get_last_report_user_count(chat_id)
            if user_count < last_report_count:
                # Messages were cleared; reset the report counter.
                last_report_count = 0
                self.store.set_last_report_user_count(chat_id, 0)
            new_count = user_count - last_report_count
            if new_count >= self.MIN_MESSAGES_FOR_REPORT:
                user_history = self.store.get_user_messages_after_count(
                    chat_id,
                    last_report_count,
                    limit=new_count,
                )
                if user_history:
                    report_language = self.store.get_report_language_for_chat(chat_id) or language
                    analysis, _ = await summarize_dialog(user_history, language=report_language)
                    self.store.add_summary(
                        chat_id=chat_id,
                        user_count=user_count,
                        summary_text=analysis.summary,
                        emotion=analysis.emotion,
                        risk_level=analysis.risk_level,
                        depression_score=analysis.depression_score,
                        anxiety_score=analysis.anxiety_score,
                        stress_score=analysis.stress_score,
                        profile_note=analysis.profile_note,
                    )
                    self.store.upsert_patient_characteristics(
                        chat_id=chat_id,
                        characteristics=analysis.key_characteristics,
                    )
                    self.store.set_last_report_user_count(chat_id, user_count)

                    if should_create_alert(analysis):
                        alert_risk_level = str(analysis.risk_level).strip().lower()
                        alert_reason = (
                            "depression_score_threshold"
                            if alert_risk_level in ("medium", "medium_high")
                            else "high_risk_summary"
                        )
                        created = self._create_alert_once(chat_id, alert_reason, analysis)
                        if (created or alert_risk_level == "high") and not crisis_prompt_sent:
                            await self._send_crisis_prompt(update, chat_id, language)
        except Exception:
            logging.exception("Error while handling incoming message")
            await update.message.reply_text(
                self._prompt_text("error", self._get_language(chat_id))
            )

    def run(self) -> None:
        """Start bot depending on delivery mode."""
        if settings.use_webhook:
            self._run_webhook()
        else:
            self._run_polling()

    def _run_polling(self) -> None:
        logging.info("Starting Telegram Bot in POLLING mode")
        self.app.run_polling()

    def _run_webhook(self) -> None:
        if not settings.webhook_host:
            raise ValueError("USE_WEBHOOK=True requires WEBHOOK_HOST to be set")
        webhook_path = (settings.webhook_path or settings.telegram_bot_token).strip("/")
        webhook_url = f"{settings.webhook_host.rstrip('/')}/{webhook_path}"
        # Do not log the URL: by default the path is the bot token.
        logging.info("Starting Telegram Bot in WEBHOOK mode on port %s", settings.webhook_port)

        self.app.run_webhook(
            listen="0.0.0.0",
            port=settings.webhook_port,
            url_path=webhook_path,
            webhook_url=webhook_url,
        )
