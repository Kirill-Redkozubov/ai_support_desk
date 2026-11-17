from dataclasses import dataclass
from collections.abc import Generator
from time import monotonic

import openai
from openai import OpenAI

from app.config import Settings
from app.llm.errors import (
LLMClientError, # главный родак ошибок епта
LLMAccessError,
LLMBalanceError,
LLMConnectionError,
LLMProviderError,
LLMRateLimitError,
LLMRequestError,
LLMTimeoutError,
)

Message = dict[str, str] # одно сообщение: {"role": ..., "content": ...}

# Технический ответ модели
@dataclass(frozen=True)
class LLMResult:
    text: str | None
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

@dataclass(frozen=True)
class LLMTextDelta:
    text: str

@dataclass(frozen=True)
class LLMStreamCompleted:
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

LLMStreamEvent = LLMTextDelta | LLMStreamCompleted

class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=str(settings.base_url),
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds, #макс время ответа модели
            max_retries=settings.max_retries, #макс попыток
        )

    def generate(self, messages: list[Message]) -> LLMResult:
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_completion_tokens=self._settings.max_output_tokens,
            )
        except openai.APIError as error:
            raise self._convert_error(error) from error

        if not response.choices: # без этой проверки [0] упал бы с IndexError
            raise LLMProviderError('Провайдер вернул ответ без вариантов.')

        choice = response.choices[0]
        usage = response.usage
        return LLMResult(
            text=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            response_id=response.id,
        )

    def stream(self, messages: list[Message]) -> Generator[LLMStreamEvent, None, None]:
        try:
            response_stream = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_completion_tokens=self._settings.max_output_tokens,
                stream=True,
                stream_options={'include_usage': True},
            )
        except openai.APIError as error:
            raise self._convert_error(error) from error

        started_at = monotonic()
        model = finish_reason = response_id = None
        prompt_tokens = completion_tokens = total_tokens = None

        try:
            for chunk in response_stream:
                if monotonic() - started_at > self._settings.stream_timeout_seconds:
                    raise LLMTimeoutError('Поток не завершился за отведенное время.')

                model = chunk.model
                response_id = chunk.id
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.completion_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                if choice.delta.content:
                    yield LLMTextDelta(text=choice.delta.content)
        except openai.APIError as error:
            raise self._convert_error(error) from error
        finally:
            response_stream.close()

        if model is None or response_id is None:
            raise LLMProviderError('Провайдер завершил поток без метаданных.')
        yield LLMStreamCompleted(
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_id=response_id,
        )


    # тут ошибки сюда ваще не смотрим они только усложняют код
    @staticmethod
    def _convert_error(error: openai.APIError) -> LLMClientError:
        """Таблица перевода: ошибка библиотеки → наша ошибка.
        Порядок важен: сначала частные случаи, потом общие."""
        request_id = getattr(error, "request_id", None)  # есть не у всех ошибок

        if isinstance(error, openai.AuthenticationError):
            return LLMAccessError("Провайдер не принял API-ключ.", request_id)
        if isinstance(error, openai.PermissionDeniedError):
            return LLMAccessError("Не хватает прав для этой модели.", request_id)
        if isinstance(error, (openai.BadRequestError, openai.NotFoundError,
                              openai.UnprocessableEntityError)):
            return LLMRequestError("Провайдер отклонил модель или параметры.", request_id)
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError("Превышен лимит запросов.", request_id)
        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError("Провайдер не ответил вовремя.")
        if isinstance(error, openai.APIConnectionError):
            return LLMConnectionError("Нет соединения с провайдером.")
        if isinstance(error, openai.APIStatusError):
            if error.status_code == 402:
                return LLMBalanceError("Не хватает денег на балансе ProxyAPI.", request_id)
            return LLMProviderError(f"Провайдер вернул HTTP {error.status_code}.", request_id)
        return LLMProviderError("Неожиданная ошибка библиотеки.")

    def close(self) -> None:
        self._client.close()