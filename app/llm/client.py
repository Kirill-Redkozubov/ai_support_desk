# для анотации, стриминга, замера времени
from dataclasses import dataclass
from collections.abc import Generator
from time import monotonic
from typing import Any, Literal

# для общения с моделью
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

# это для ошибок и получения ключей
from app.config import Settings
from app.llm.errors import (
    LLMClientError,  # главный родак ошибок епта
    LLMAccessError,
    LLMBalanceError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
)

# структура для одного сообщения user, developer или assistant
Message = dict[str, Any]  # одно сообщение: {"role": ..., "content": ...}
JsonSchema = dict[str, Any]
ToolDefinition = dict[str, Any]

# вызов инструмента моделью
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


# Технический ответ модели
@dataclass(frozen=True)
class LLMResult:
    text: str | None
    refusal: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


# эти два класса для анотации стриминга
# это для чанки
@dataclass(frozen=True)
class LLMTextDelta:
    text: str


# это для результата стриминга
@dataclass(frozen=True)
class LLMStreamCompleted:
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


# это анотация для вывода stream
LLMStreamEvent = LLMTextDelta | LLMStreamCompleted


# для запросов и ответов с моделью только этот класс
class LLMClient:
    # для создания эк с настройками и клиентом для общения
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=str(settings.base_url),
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,  # макс время ответа модели
            max_retries=settings.max_retries,  # макс попыток
        )

    # нужен только для сжатия самори тут стриминга нет
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

        return self._to_result(response)

    # для получения ответа в json
    def generate_structured(
            self,
            messages: list[Message],
            schema_name: str,
            schema: JsonSchema,
    ) -> LLMResult:
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_completion_tokens=self._settings.max_output_tokens,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'strict': True,
                        'schema': schema,
                    }
                }
            )
        except openai.APIError as error:
            raise self._convert_error(error) from error

        return self._to_result(response)

    # обычный текст или tool call
    def generate_with_tools(
            self,
            messages: list[Message],
            tools: list[ToolDefinition], # dict[str, Any] схема функции
            tool_choice: Literal['auto', 'none'] = 'auto'
    ) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
        )
        return self._to_result(response)


    # нужен для стриминга в чат клиенту
    def stream(self, messages: list[Message]) -> Generator[LLMStreamEvent, None, None]:
        # открываем соединение для стриминга
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

        # время и метрики сразу None чтобы не было ошибки в конце
        started_at = monotonic()
        model = finish_reason = response_id = None
        prompt_tokens = completion_tokens = total_tokens = None

        # читаем чанки на каждом цикле из открытого соединения
        try:
            for chunk in response_stream:
                if monotonic() - started_at > self._settings.stream_timeout_seconds:
                    raise LLMTimeoutError('Поток не завершился за отведенное время.')
                # для вывода конечных метрикиков
                model = chunk.model
                response_id = chunk.id
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
                if not chunk.choices:
                    continue
                # для вывода чанка или причины
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                if choice.delta.content:
                    yield LLMTextDelta(text=choice.delta.content)
        except openai.APIError as error:
            raise self._convert_error(error) from error
        finally:
            response_stream.close()
        # конец для отправки метриков
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

    # чтобы не дублировать везде длинный вывод LLMResul
    @staticmethod
    def _to_result(response: ChatCompletion) -> LLMResult:
        if not response.choices:
            raise LLMProviderError('Провайдер вернул ответ без вариантов продолжения.')

        choice = response.choices[0]
        usage = response.usage
        tool_calls = tuple(
            ToolCall(
                id=tool_call.id, # str
                name=tool_call.function.name, # str
                arguments=tool_call.function.arguments, # json строка '{'order_id': '5643'}'
            )
            for tool_call in (choice.message.tool_calls or [])
        )
        return LLMResult(
            text=choice.message.content,
            refusal=choice.message.refusal,
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            response_id=response.id,
        )

    # для обработки ошибок
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
