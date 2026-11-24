# для анотации результата стриминга ответа, для блокировки потока, для замера времени
from dataclasses import dataclass, field
from collections.abc import Generator
from threading import Lock
from time import perf_counter

# для запроса к ллм, анотации и обработки ошибок
from app.llm.client import Message
from app.services.tool_loop import BoundedToolLoop, ToolLoopError

# для расчета токенов,
from app.chat_history import ChatHistoryError, SlidingWindowHistory
from app.services.conversation_summary_service import ConversationSummaryError, ConversationSummaryService

# для поведения модели основных запросов
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT
from app.prompts.template import PromptTemplate


# для обработки ошибок при обращении к моделе
class ChatServiceError(RuntimeError): ...


# анотация метриков и ответа от модели
@dataclass(frozen=True)
class ChatReply:
    text: str
    session_id: str
    history_messages: int  # сколько сообщений в истории после этого хода
    context_history_messages: int
    removed_history_messages: int
    estimated_prompt_tokens: int
    summary_updated: bool
    has_summary: bool
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    response_ids: tuple[str, ...]
    tool_calls_count: int
    tool_names: tuple[str, ...]


# для чанка
@dataclass(frozen=True)
class ChatTextDelta:
    text: str


# для анотации другой блядь анотации
@dataclass(frozen=True)
class ChatStreamCompleted:
    result: ChatReply  # результат после чанков


ChatStreamEvent = ChatTextDelta | ChatStreamCompleted


# набор для хранения списка всей истории сообщений, строки с сжатой историей этого же списка
@dataclass
class ChatSession:
    messages: list[Message] = field(default_factory=list)  # диалог
    summary: str | None = None  # сжатый диалог, краткий пересказ messages от модели
    lock: Lock = field(default_factory=Lock)  # и своя блокировка


class ChatService:
    def __init__(self,
                 tool_loop: BoundedToolLoop,
                 history_policy: SlidingWindowHistory,  # инструмент
                 summary_service: ConversationSummaryService,
                 prompt: PromptTemplate = SUPPORT_CHAT_PROMPT) -> None:  # инструмент
        self._tool_loop = tool_loop
        self._history_policy = history_policy  # для расчета в токенах и обреза session.messages
        self._summary_service = summary_service  # для создания нового session.summary
        self._prompt = prompt
        self._chat_instruction = prompt.render()
        self._sessions: dict[str, ChatSession] = {}
        self._sessions_lock = Lock()

    # для быстрой проверки evals
    def reply(self, session_id: str, user_text: str) -> ChatReply:
        for event in self.stream_reply(session_id, user_text):
            if isinstance(event, ChatStreamCompleted):
                return event.result
        raise ChatServiceError('Поток завершился без итогового результата.')

    def stream_reply(self, session_id: str, user_text: str) -> Generator[ChatStreamEvent, None, None]:
        session_key = self._normalize_session_id(session_id)  # вместо .strip()

        # для создания нового или поиск существующего ChatSession()
        session = self._get_or_create_session(session_key)
        text = user_text.strip()
        if not text:
            raise ChatServiceError('Сообщение не должно быть пустым.')

        # замок чтобы новый ChatSession() не мог отправить два сообщения подряд
        with session.lock:
            developer_message: Message = {'role': 'developer', 'content': self._chat_instruction}
            user_message: Message = {
                'role': 'user',
                'content': f'<customer_message>\n{text}\n</customer_message>',
            }

            # [] | [Message] проверка что session.messages проходит полностью по токенам
            summary_message = [self._summary_message(session.summary)] if session.summary else []
            try:
                prepared = self._history_policy.prepare(  # обрезанный/тот же list[Message] и метрики
                    developer_message=developer_message,
                    history=session.messages,  # [] | list[Message]
                    user_message=user_message,
                    summary_message=summary_message,  # [Message] | []
                )
            except ChatHistoryError as error:
                raise ChatServiceError(f'Не удалось подготовить историю: {error}') from error

            started_at = perf_counter()
            new_summary: str | None = None

            # если непроходит и в prepared обрезанный list[Message]
            if prepared.removed_history_messages > 0:
                try:
                    new_summary = self._summary_service.summarize(  # str новый краткий пересказ от модели
                        previous_summary=session.summary,  # str
                        messages=session.messages,  # list[Message]
                    )
                except ConversationSummaryError as error:  # ловим чужую ошибку
                    # выводим нашу: чужую ошибку
                    raise ChatServiceError(f'Не удалось сократить историю: {error}') from error
                try:
                    prepared = self._history_policy.prepare(  # повторяем проверку но толкьо с пересказом
                        developer_message=developer_message,
                        history=[],
                        user_message=user_message,
                        summary_message=[self._summary_message(new_summary)],
                    )
                except ChatHistoryError as error:
                    raise ChatServiceError(f'Новое summary не помещается в бюджет: {error}') from error
            # новый кусок сервис теперь работает через BoundedToolLoop, а не client напрямую
            try:
                loop_result = self._tool_loop.run(prepared.messages)
            except ToolLoopError as error:
                raise ChatServiceError(
                    f"Не удалось получить ответ: {error}"
                ) from error

            elapsed_seconds = perf_counter() - started_at
            reply_text = ''.join(loop_result.text).strip()
            if not reply_text:
                raise ChatServiceError("Модель вернула пустой ответ.")

            yield ChatTextDelta(text=reply_text)

            # обновляем session.summary и session.messages
            assistant_message: Message = {'role': 'assistant', 'content': reply_text}
            if new_summary is not None:
                session.summary = new_summary
                session.messages = [user_message, assistant_message]
            else:
                session.messages += [user_message, assistant_message]

            result = ChatReply(
                text=reply_text,
                session_id=session_key,
                history_messages=len(session.messages),
                context_history_messages=prepared.context_history_messages,
                removed_history_messages=prepared.removed_history_messages,
                estimated_prompt_tokens=prepared.estimated_tokens,
                summary_updated=new_summary is not None,
                has_summary=session.summary is not None,
                model=loop_result.model,
                prompt_id=self._prompt.prompt_id,
                prompt_version=self._prompt.version,
                elapsed_seconds=elapsed_seconds,
                prompt_tokens=loop_result.prompt_tokens,
                completion_tokens=loop_result.completion_tokens,
                total_tokens=loop_result.total_tokens,
                response_id=loop_result.response_ids[-1],
                response_ids=loop_result.response_ids,
                tool_calls_count=loop_result.tool_calls_count,
                tool_names=loop_result.tool_names,
            )
        yield ChatStreamCompleted(result=result)

    # for delete list[Message] from session
    def reset_session(self, session_id: str) -> None:
        session_key = self._normalize_session_id(session_id)
        with self._sessions_lock:
            session = self._sessions.get(session_key)
        if session is None:
            return
        with session.lock:
            session.messages.clear()
            session.summary = None

    # метод для получения или создания нового эк ChatSession()
    def _get_or_create_session(self, session_key: str) -> ChatSession:
        with self._sessions_lock:
            session = self._sessions.get(session_key)
            if session is None:
                session = ChatSession()
                self._sessions[session_key] = session
            return session

    # целый отдельный мать его метод для простого strip()
    # причем пустой session_id сюда никак не придет
    # тупо проверка ради проверки
    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        session_key = session_id.strip()
        if not session_key:
            raise ChatServiceError('Session ID не должен быть пустым.')
        return session_key

    # из строки summary сделать словарь мессадж
    @staticmethod
    def _summary_message(summary: str) -> Message:
        return {
            'role': 'user',
            'content': f'<conversation_summary>\n{summary}\n</conversation_summary>'
        }
