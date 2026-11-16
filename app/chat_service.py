from dataclasses import dataclass
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError

from app.prompts.support_chat import SUPPORT_CHAT_PROMPT
from skeleton import user_message

CHAT_INSTRUCTION = SUPPORT_CHAT_PROMPT.render()

class ChatServiceError(RuntimeError): ...

@dataclass(frozen=True)
class ChatReply:
    text: str
    history_messages: int          # сколько сообщений в истории после этого хода
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._history: list[Message] = [] # list[dict]

    def reply(self, user_text: str) -> ChatReply:
        text = user_text.strip()
        if not text:
            raise ChatServiceError('Сообщение не должно быть пустым.')

        user_message: Message = {
            'role': 'user',
            'content': f'<customer_message>\n{text}\n</customer_message>',
        }
        messages = [
            {'role': 'developer', 'content': CHAT_INSTRUCTION},
            *self._history,
            user_message,
        ]

        started_at = perf_counter()
        try:
            # ////// запрос и ответ от модели
            result = self._llm_client.generate(messages)
        except LLMClientError as error:
            raise ChatServiceError(f"Не удалось получить ответ модели: {error}") from error
        elapsed = perf_counter() - started_at

        if result.finish_reason == "length":
            raise ChatServiceError("Ответ обрезан по лимиту длины.")
        if result.finish_reason != "stop":
            raise ChatServiceError(f"Модель не дала готовый ответ: {result.finish_reason}.")
        reply_text = (result.text or "").strip()
        if not reply_text:
            raise ChatServiceError("Модель вернула пустой ответ.")
        # += это extend он распаковывает, append вкладывает список целиком
        self._history += [user_message, {'role': 'assistant', 'content': reply_text}]

        return ChatReply(
            text=reply_text,
            history_messages=len(self._history),
            model=result.model,
            prompt_id=SUPPORT_CHAT_PROMPT.prompt_id,
            prompt_version=SUPPORT_CHAT_PROMPT.version,
            elapsed_seconds=elapsed,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            response_id=result.response_id,
        )























