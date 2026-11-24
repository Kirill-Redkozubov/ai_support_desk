"""Эксперимент: тот же диалог, что в try_responses.py, но через наш ChatService
(скользящее окно + summary). Считаем все запросы, включая запросы на summary.

Не часть проекта. Запуск: python try_our_way.py
"""

import services.chat_service as chat_service_module
from app.chat_history import SlidingWindowHistory, TokenCounter
from services.chat_service import ChatService
from app.config import Settings
from services.conversation_summary_service import ConversationSummaryService
from app.llm.client import LLMClient, LLMStreamCompleted

# всё как в try_responses.py
INSTRUCTION = "Ты отвечаешь клиенту поддержки интернет-магазина. Отвечай кратко, 1-2 предложения."
THRESHOLD = 1500  # у нас это бюджет окна
FILLER = "Ещё подробности: " + "курьер опоздал на два часа, коробка мятая, скотч переклеен. " * 30

TURNS = [
    "Здравствуйте. Номер моего заказа 7718, монитор пришёл с треснутой подставкой.",
    FILLER,
    FILLER,
    FILLER,
    "Напомните, какой номер заказа я назвал в самом начале?",
]

# подменяем промпт поддержки из курса на тот же короткий, что в try_responses.py
chat_service_module.CHAT_INSTRUCTION = INSTRUCTION


class CountingLLMClient(LLMClient):
    """Наш клиент, который дополнительно записывает usage каждого запроса."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls: list[tuple[str, int, int]] = []  # (что за запрос, вход, выход)

    def generate(self, messages):
        result = super().generate(messages)  # generate у нас вызывает только summary
        self.calls.append(("summary", result.prompt_tokens or 0, result.completion_tokens or 0))
        return result

    def stream(self, messages):
        for event in super().stream(messages):
            if isinstance(event, LLMStreamCompleted):
                self.calls.append(("ответ", event.prompt_tokens or 0, event.completion_tokens or 0))
            yield event


settings = Settings().model_copy(update={"max_output_tokens": 1000})  # как в try_responses.py
llm_client = CountingLLMClient(settings)
chat_service = ChatService(
    llm_client,
    history_policy=SlidingWindowHistory(token_counter=TokenCounter(), input_token_budget=THRESHOLD),
    summary_service=ConversationSummaryService(llm_client),
)

try:
    for number, text in enumerate(TURNS, start=1):
        calls_before = len(llm_client.calls)
        reply = chat_service.reply("try-our-way", text)
        for kind, prompt_tokens, completion_tokens in llm_client.calls[calls_before:]:
            print(f"Ход {number} [{kind}]: вход={prompt_tokens} выход={completion_tokens}")
        print(f"  Ответ: {reply.text}")
finally:
    llm_client.close()

total_in = sum(call[1] for call in llm_client.calls)
total_out = sum(call[2] for call in llm_client.calls)
print(f"\nНаш код:                вход={total_in} выход={total_out} в пересчёте на входные={total_in + 4 * total_out}")
print("Responses + compaction: вход=8877 выход=1774 в пересчёте на входные=15973 (твой прошлый замер)")
