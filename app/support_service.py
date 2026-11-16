from dataclasses import dataclass
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError

from app.prompts.support_summary import SUPPORT_SUMMARY_PROMPT

DEVELOPER_INSTRUCTION = SUPPORT_SUMMARY_PROMPT.render(response_language="русском")
MAX_SUMMARY_CHARS = 300


class SupportServiceError(RuntimeError): ...

@dataclass(frozen=True)
class SupportSummary:
    """Вывод/результат summarize"""
    text: str
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

class SupportService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client # готовый клиент приходит снаружи

    def summarize(self, user_text: str) -> SupportSummary:
        text = user_text.strip()
        if not text:
            raise SupportServiceError('Обращение не должно быть пустым.')

        started_at = perf_counter()
        try:
            result = self._llm_client.generate(self._build_messages(text))
        # from error нужен чтобы типо в выводе вся цепочка обработанных eroorov была
        # типо вывод SupportServiceError: LLMClientError
        # тут опять кароче лапша с ошибками которую как по мне лучше вынести
        # чтобы тут код можно было адекватно читать
        except LLMClientError as error:
            raise SupportServiceError(
            f'Не удалось получить ответ модели: {error}'
        ) from error
        elapsed = perf_counter() - started_at

        if result.finish_reason == "length":
            raise SupportServiceError("Ответ модели обрезан по лимиту длины.")
        if result.finish_reason != "stop":
            raise SupportServiceError(
                f"Модель не дала готовый ответ: {result.finish_reason}."
            )

        return SupportSummary(
            text=self._validate_summary(result.text),
            model=result.model,
            prompt_id=SUPPORT_SUMMARY_PROMPT.prompt_id,
            prompt_version=SUPPORT_SUMMARY_PROMPT.version,
            elapsed_seconds=elapsed,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            response_id=result.response_id,
        )

    @staticmethod
    def _build_messages(user_text: str) -> list[Message]:
        return [
            {'role': 'system', 'content': DEVELOPER_INSTRUCTION},
            {'role': 'user', 'content': f"<customer_request>\n{user_text}\n</customer_request>"},
        ]

    # просто валидация ответа модели, возвращает strip() ответа
    @staticmethod
    def _validate_summary(summary: str | None) -> str:
        if summary is None:
            raise SupportServiceError("Модель не вернула текст.")
        cleaned = summary.strip()
        if not cleaned:
            raise SupportServiceError("Модель вернула пустой ответ.")
        if len(cleaned) > MAX_SUMMARY_CHARS:
            raise SupportServiceError("Резюме слишком длинное.")
        return cleaned


































