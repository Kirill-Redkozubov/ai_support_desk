from dataclasses import dataclass

# для расчета в токенах
import tiktoken

# импорт анотации для одного сообщения {'role': ..., 'content': ...}
from app.llm.client import Message

# для первоначального примерного расчета в токенах
TOKENS_PER_MESSAGE = 3
REPLY_PRIMER_TOKENS = 3


class ChatHistoryError(RuntimeError): ...


# анотация результата prepare()
@dataclass(frozen=True)
class PreparedHistory:
    messages: list[Message]
    estimated_tokens: int
    context_history_messages: int
    removed_history_messages: int


class TokenCounter:
    def __init__(self, encoding_name: str = 'o200k_base') -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    # считает каждый {'role':..., 'content'} в словаре и возвращает число - сумму в токенах
    def count_messages(self, messages: list[Message]) -> int:
        total = REPLY_PRIMER_TOKENS
        for message in messages:
            total += TOKENS_PER_MESSAGE
            total += len(self._encoding.encode(message['role']))
            total += len(self._encoding.encode(message['content']))
        return total


class SlidingWindowHistory:
    def __init__(self, token_counter: TokenCounter, input_token_budget: int) -> None:
        self._token_counter = token_counter
        self._budget = input_token_budget

    def prepare(
            self,
            developer_message: Message,
            history: list[Message],  # session.message
            user_message: Message,
            summary_message: list[Message],  # [Message]
    ) -> PreparedHistory:
        turns = self._split_into_turns(history)  # list[list[Message]]

        # получаем int - стоимость всего контекста и сравниваем с бюджетом
        required = [developer_message, *summary_message, user_message]
        if self._token_counter.count_messages(required) > self._budget:
            raise ChatHistoryError('Инструкция, summary и новое сообщение не помещаются в бюджет.')

        selected: list[Message] = []  # заполняем пока влезает из session.messages
        for turn in reversed(turns):
            candidate = [*turn, *selected]
            tokens = self._token_counter.count_messages(
                [developer_message, *summary_message, *candidate, user_message]
            )
            if tokens > self._budget:
                break
            selected = candidate

        messages = [developer_message, *summary_message, *selected, user_message]
        return PreparedHistory(
            messages=messages,
            estimated_tokens=self._token_counter.count_messages(messages),
            context_history_messages=len(selected),
            removed_history_messages=len(history) - len(selected),
        )

    # для склейки на диалоговые пары внутри списка
    # [['role':'user', 'content':...],['role':'assistant', 'content':...]]
    @staticmethod
    def _split_into_turns(history: list[Message]) -> list[list[Message]]:
        if len(history) % 2 != 0:
            raise ChatHistoryError('История должна состоять из законченных ходов.')
        turns = [history[i:i + 2] for i in range(0, len(history), 2)]
        for turn in turns:
            if [message['role'] for message in turn] != ['user', 'assistant']:
                raise ChatHistoryError('Ожидалась пара сообщений user/assistant.')
        return turns
