"""HTTPS webhook приватного навыка Алисы для Vercel.
Gemini Interactions API, стандартная библиотека Python.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
MAX_ANSWER_CHARS = 900
TIME_BUDGET_SECONDS = 3.65

INSTRUCTIONS = (
    "Ты голосовой помощник в приватном навыке Алисы. Отвечай на русском языке "
    "естественно и кратко: обычно одно-два предложения, до 40 слов. Пиши только текст "
    "для произнесения вслух, без Markdown, списков, ссылок и эмодзи. Если данных "
    "недостаточно, честно скажи об этом. Учитывай предыдущие сообщения этой беседы. "
    "Не заявляй, что выполнил действие во внешнем сервисе, если оно не было выполнено."
)
SEARCH_PROMPT = (
    "Используй Google Search для проверки актуальных сведений. Ответь на русском кратко, "
    "только по найденным данным. Назови источник обычными словами, не произноси адреса сайтов. "
    "Если подтверждения нет, прямо скажи об этом."
)
SEARCH_FOLLOWUP = ("что нашлось", "что нашёл", "ну что нашёл", "ну что нашлось")


class GeminiError(Exception):
    """Ошибка обращения к Gemini API."""


def _api_request(method, url, key, payload=None, timeout=2.5):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {
        "x-goog-api-key": key,
        "Content-Type": "application/json",
        "Api-Revision": "2026-05-20",
    }
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            error = detail.get("error", {})
            message = str(error.get("message") or "")
            status = str(error.get("status") or error.get("code") or "unknown")
            logging.warning("Gemini API HTTP %s (%s): %s", exc.code, status, message[:300])
        except (OSError, ValueError, AttributeError):
            status, message = "unknown", ""
            logging.warning("Gemini API HTTP %s", exc.code)

        if exc.code == 400:
            raise GeminiError("Gemini отклонил запрос. Проверьте настройки модели.") from exc
        if exc.code == 401 or exc.code == 403:
            raise GeminiError("Ключ Gemini API не принят или у него нет доступа.") from exc
        if exc.code == 404:
            raise GeminiError("Запрошенный ответ или модель Gemini не найдены.") from exc
        if exc.code == 429:
            raise GeminiError("Сейчас достигнут бесплатный лимит Gemini API.") from exc
        raise GeminiError("Сервис Gemini временно недоступен.") from exc
    except (OSError, ValueError) as exc:
        logging.warning("Gemini API request failed: %s", type(exc).__name__)
        raise GeminiError("Не удалось связаться с Gemini.") from exc


def _needs_web_search(command):
    if re.search(r"\b(найди|поищи|ищи|загугли|погугли)\b", command):
        return True
    if re.search(r"\b(в интернете|в сети|по интернету)\b", command):
        return True
    if re.search(r"\b(новости|новостей|погода|погоду|прогноз погоды)\b", command):
        return True
    return command.startswith(("что нового", "что произошло сегодня", "какой сейчас курс"))


def _answer_text(data):
    parts = []
    for step in data.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    text = " ".join(parts)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s*\((?:[\w-]+\.)+[\w-]{2,}(?:/[^)]*)?\)", "", text)
    text = " ".join(text.split()).strip()
    if len(text) > MAX_ANSWER_CHARS:
        cut = text[:MAX_ANSWER_CHARS]
        sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[:sentence_end + 1] if sentence_end > 400 else cut.rstrip() + "…"
    return text or "Ответ пока не получился. Попробуйте спросить иначе."


def _say(text, state=None, end_session=False):
    result = {
        "version": "1.0",
        "response": {"text": text[:1024], "end_session": end_session},
    }
    if state is not None and not end_session:
        result["session_state"] = state
    return result


def _interaction_url(interaction_id):
    return API_URL + "/" + urllib.parse.quote(interaction_id, safe="")


def _finish(data):
    answer = _answer_text(data)
    interaction_id = data.get("id", "")
    state = {"previous_interaction_id": interaction_id} if interaction_id else {}
    return _say(answer, state)


def _valid_interaction_id(value):
    return isinstance(value, str) and value.startswith("v1_") and len(value) < 200


def handler_alice(event, context):
    """Обработчик события Яндекс Диалогов."""
    del context
    start = time.monotonic()

    if not isinstance(event, dict):
        return _say("Не удалось прочитать запрос. Повторите, пожалуйста.")

    req = event.get("request") or {}
    utterance = (req.get("command") or req.get("original_utterance") or "").strip()
    command = utterance.casefold().strip(" .,!?;:")
    state = (event.get("state") or {}).get("session") or {}
    if not isinstance(state, dict):
        state = {}

    previous_id = state.get("previous_interaction_id", "")
    if not _valid_interaction_id(previous_id):
        previous_id = ""

    if command in ("хватит", "выход", "закончить", "стоп"):
        return _say("До встречи!", end_session=True)
    if command in ("сброс", "забудь разговор", "начать заново"):
        return _say("Начинаем новый разговор. Что хотите спросить?", {})
    if re.search(r"\b(поставь|установи|заведи|включи|создай)\b.*\b(будильник|таймер|напоминание)\b", command):
        return _say(
            "Системный будильник, таймер или напоминание я из навыка не устанавливаю. "
            "Скажите «выход», затем обратитесь с просьбой к Алисе.",
            state,
        )

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return _say("В навыке ещё не настроен ключ Gemini API.")

    pending_id = state.get("pending_interaction_id", "")
    if pending_id:
        if not _valid_interaction_id(pending_id):
            return _say("Не удалось продолжить ответ. Повторите вопрос.", {})
        is_search = bool(state.get("pending_search"))
        waiting_text = (
            "Ещё ищу. Спросите «что нашлось?» через несколько секунд."
            if is_search else "Ещё думаю. Скажите «готово» через несколько секунд."
        )
        try:
            data = _api_request("GET", _interaction_url(pending_id), key, timeout=2.2)
        except GeminiError as exc:
            return _say(str(exc) + " " + waiting_text, state)

        status = data.get("status")
        if status == "completed":
            answer = _finish(data)
            if command not in ("", "готово", "ответ", "дальше", "ну что", *SEARCH_FOLLOWUP):
                answer["response"]["text"] += " Повторите ваш новый вопрос."
            return answer
        if status in ("in_progress", "queued"):
            return _say(waiting_text, state)
        return _say("Получить ответ не удалось. Повторите вопрос.", {})

    if not utterance or command in ("готово", "ответ", "дальше", *SEARCH_FOLLOWUP):
        return _say("Здравствуйте! Задайте вопрос, и я постараюсь ответить.", state)

    search = _needs_web_search(command)
    payload = {
        "model": MODEL,
        "input": utterance[:1000],
        "system_instruction": INSTRUCTIONS + (" " + SEARCH_PROMPT if search else ""),
        "background": True,
        "store": True,
        "generation_config": {
            "temperature": 0.4,
            "thinking_level": "low",
            "max_output_tokens": 220 if search else 120,
        },
    }
    if search:
        payload["tools"] = [{"type": "google_search"}]
    if previous_id:
        payload["previous_interaction_id"] = previous_id

    try:
        data = _api_request("POST", API_URL, key, payload=payload, timeout=2.8)
        if data.get("status") == "completed":
            return _finish(data)

        interaction_id = data.get("id", "")
        if not _valid_interaction_id(interaction_id):
            logging.warning("Gemini returned unexpected interaction response: %s", str(data)[:500])
            return _say("Получить ответ не удалось. Повторите вопрос.", state)

        # Проверяем готовность в пределах времени, доступного навыку Алисы.
        url = _interaction_url(interaction_id)
        while (remaining := TIME_BUDGET_SECONDS - (time.monotonic() - start)) > 0.75:
            try:
                ready = _api_request("GET", url, key, timeout=min(remaining - 0.2, 0.75))
                if ready.get("status") == "completed":
                    return _finish(ready)
                if ready.get("status") not in ("queued", "in_progress"):
                    break
            except GeminiError:
                break
            remaining = TIME_BUDGET_SECONDS - (time.monotonic() - start)
            if remaining > 0.95:
                time.sleep(min(0.18, remaining - 0.75))

        return _say(
            "Ищу информацию. Спросите «что нашлось?» через несколько секунд."
            if search else "Думаю над ответом. Скажите «готово» через несколько секунд.",
            {"pending_interaction_id": interaction_id, "pending_search": search},
        )
    except GeminiError as exc:
        return _say(str(exc) + " Повторите вопрос позже.", state)


# HTTPS-вебхук Яндекс Диалогов для Vercel.
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit
import hmac

SKILL_ID = os.environ.get("ALICE_SKILL_ID", "").strip()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        expected = os.environ.get("ALICE_WEBHOOK_TOKEN", "")
        supplied = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
        if not expected or not hmac.compare_digest(supplied, expected):
            return self._reply(404, {"error": "not found"})
        if not SKILL_ID:
            return self._reply(503, {"error": "ALICE_SKILL_ID is not configured"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16384:
                return self._reply(413, {"error": "invalid request size"})
            event = json.loads(self.rfile.read(length))
            if not isinstance(event, dict) or (event.get("session") or {}).get("skill_id") != SKILL_ID:
                return self._reply(403, {"error": "invalid skill"})
            return self._reply(200, handler_alice(event, None))
        except (ValueError, json.JSONDecodeError):
            return self._reply(400, {"error": "invalid JSON"})

    def do_GET(self):
        self._reply(405, {"error": "method not allowed"})

    def _reply(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # Не записывать секрет вебхука из URL в журнал приложения.
