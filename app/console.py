from uuid import uuid4

from app.services.chat_service import (
    ChatReply,
    ChatServiceError,
    ChatService,
    ChatStreamCompleted,
    ChatTextDelta,
)
from app.services.support_request_analyzer import (
SupportRequestAnalyzer,
SupportRequestAnalyzerError,
)
from app.services.support_request_review import (
SupportRequestDecision,
decide_review,
)
import readline

# # для метриков результата вызова инструментов tool_call
# def print_order_status_answer(result: OrderStatusAnswer) -> None:
#     print(f'\nAssistant: {result.text}')
#     print(
#         'Инструмент использован: '
#         f'{'да' if result.tool_used else 'нет'}'
#     )
#     if result.order_id is not None:
#         print(f'Номер заказа: {result.order_id}')
#     print(f'Промпт: {result.prompt_id}@{result.prompt_version}')
#     print(f'Всего токенов: {result.total_tokens}')
#     print(
#         'ID ответов: '
#         f'{', '.join(result.response_ids)}'
#     )



def print_metadata(result: ChatReply) -> None:
    print(f"Session ID ответа: {result.session_id}")
    print(f"Сообщений в истории: {result.history_messages}")
    print(f"Сообщений истории в запросе: {result.context_history_messages}")
    print(f"Сообщений истории отброшено: {result.removed_history_messages}")
    print(f"Оценка входных токенов до запроса: {result.estimated_prompt_tokens}")
    print(f"Summary сохранено: {'да' if result.has_summary else 'нет'}")
    print(f"Summary обновлено на этом ходу: {'да' if result.summary_updated else 'нет'}")
    print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
    print(f"Время ответа: {result.elapsed_seconds:.2f} с")
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
    else:
        print(f"Токены: вход {result.prompt_tokens}, "
              f"выход {result.completion_tokens}, всего {result.total_tokens}")
    print(f"Инструментов выполнено: {result.tool_calls_count}")
    if result.tool_names:
        print(f"Инструменты: {', '.join(result.tool_names)}")
    print(f"ID ответов: {', '.join(result.response_ids)}")

def print_analysis(decision: SupportRequestDecision) -> None:
    result = decision.result

    print('\nАнализ обращения:')
    print(result.analysis.model_dump_json(indent=2))
    print(
        'Нужна проверка человеком: '
        f'{'да' if decision.requires_human_review else 'нет'}'
    )

    if decision.review_reasons:
        reasons = ', '.join(reason.value for reason in decision.review_reasons)
        print(f'Причины проверки: {reasons}')

    print(f'Промпт: {result.prompt_id}@{result.prompt_version}')
    print(f'Исправляющих запросов: {result.repair_attempts}')
    print(f'Время анализа: {result.elapsed_seconds:.2f} c')
    print(f'Всего токенов: {result.total_tokens}')
    print(f'ID ответа: {result.response_id}')


def run_console(service: ChatService,
                analyzer: SupportRequestAnalyzer,
                app_env: str,
                configured_model: str,) -> None:
    print(f'Окружение: {app_env}')
    print(f'Настроенная модель: {configured_model}')

    session_id = uuid4().hex
    print(f'Session ID: {session_id}')
    print(
        "Команды: /new — новый диалог, "
        "/analyze — анализ обращения, "
        "/exit — выход."
    )
    print('Во время ответа нажмите Ctrl+C, тобы отменить генерацию.')

    while True:
        user_text = input('\nВы: ')
        command = user_text.strip()
        if command == '/exit':
            return
        if command == '/new':
            service.reset_session(session_id)
            session_id = uuid4().hex
            print(f'Новый Session ID: {session_id}')
            continue
        if command == '/analyze':
            request_text = input('Обращение: ')
            try:
                analysis_result = analyzer.analyze(request_text)
            except SupportRequestAnalyzerError as error:
                print(f'Ошибка анализа: {error}')
                continue
            decision = decide_review(analysis_result)
            print_analysis(decision)
            continue

        stream = service.stream_reply(session_id, user_text)
        result: ChatReply | None = None
        received_text = False

        print('Assistant: ', end='', flush=True)
        try:
            for event in stream:
                if isinstance(event, ChatTextDelta):
                    received_text = True
                    print(event.text, end='', flush=True)
                elif isinstance(event, ChatStreamCompleted):
                    result = event.result
        except KeyboardInterrupt:
            stream.close()
            print('\nГенерация отменена. Ответ не сохранен в истории.')
            continue
        except ChatServiceError as error:
            stream.close()
            print()
            print(f'Поток прерван: {error}' if received_text else f'Ошибка чата: {error}')
            continue
        print()

        if result is None:
            print('Ошибка чата: поток завершился без результата.')
            continue
        print_metadata(result)






























