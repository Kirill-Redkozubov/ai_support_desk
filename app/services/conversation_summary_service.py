from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError
from app.prompts.chat_history_summary import CHAT_HISTORY_SUMMARY_PROMPT

SUMMARY_INSTRUCTION = CHAT_HISTORY_SUMMARY_PROMPT.render()
ROLE_NAMES = {'user': 'Клиент', 'assistant': 'Поддержка'}

class ConversationSummaryError(RuntimeError): ...


class ConversationSummaryService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def summarize(self, previous_summary: str | None, messages: list[Message]) -> str:
        if not messages:
            raise ConversationSummaryError('Нет сообщений для суммаризации')

        transcript = '\n'.join( # user меняем на Клиент, assistant на Поддержка
            f'{ROLE_NAMES.get(m['role'], m['role'])}: {m['content']}' for m in messages
        )
        source = ( # добавляем старую summary если есть
            f'<previous_summary>\n{previous_summary or 'Отсутствует.'}\n</previous_summary>\n\n'
            f'<conversation_fragment>\n{transcript}\n</conversation_fragment>'
        )
        request: list[Message] = [ # собираем промпт
            {'role': 'developer', 'content': SUMMARY_INSTRUCTION},
            {'role': 'user', 'content': source}
        ]

        try:
            result = self._llm_client.generate(request) # запрос к модели для пересказа
        except LLMClientError as error:
            raise ConversationSummaryError(f"Не удалось создать summary: {error}") from error

        if result.finish_reason == "length":
            raise ConversationSummaryError("Summary обрезано по лимиту длины.")
        if result.finish_reason != "stop":
            raise ConversationSummaryError(f"Модель не дала готовое summary: {result.finish_reason}.")
        summary = (result.text or "").strip()
        if not summary:
            raise ConversationSummaryError("Модель вернула пустое summary.")
        return summary # str новый краткий пересказ
