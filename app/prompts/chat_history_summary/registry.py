from app.prompts.chat_history_summary.v1_0_0 import CHAT_HISTORY_SUMMARY_PROMPT_V1_0_0
from app.prompts.chat_history_summary.v1_1_0 import CHAT_HISTORY_SUMMARY_PROMPT_V1_1_0
from app.prompts.template import PromptTemplate, PromptTemplateError

ACTIVE_CHAT_HISTORY_SUMMARY_VERSION = "1.1.0"

_PROMPTS: dict[str, PromptTemplate] = {
    prompt.version: prompt for prompt in (CHAT_HISTORY_SUMMARY_PROMPT_V1_0_0, CHAT_HISTORY_SUMMARY_PROMPT_V1_1_0)
}


def get_chat_history_summary_prompt(version: str | None = None) -> PromptTemplate:
    selected = ACTIVE_CHAT_HISTORY_SUMMARY_VERSION if version is None else version.strip()
    if selected not in _PROMPTS:
        raise PromptTemplateError(
            f"Неизвестная версия chat_history_summary: {selected}. "
            f"Доступны: {', '.join(_PROMPTS)}."
        )
    return _PROMPTS[selected]