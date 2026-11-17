from pydantic import ValidationError

from app.chat_history import SlidingWindowHistory, TokenCounter
from app.conversation_summary_service import ConversationSummaryService
from app.config import Settings
from app.console import run_console
from app.llm.client import LLMClient
from app.chat_service import ChatService


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    llm_client = LLMClient(settings)
    history_policy = SlidingWindowHistory(
        token_counter=TokenCounter(),
        input_token_budget=settings.chat_input_token_budget,
    )
    summary_service = ConversationSummaryService(llm_client)
    chat_service = ChatService(
        llm_client,
        history_policy=history_policy,
        summary_service=summary_service,
    )
    try:
        run_console(chat_service, app_env=settings.app_env, configured_model=settings.model)
    finally:
        llm_client.close()


if __name__ == '__main__':
    main()