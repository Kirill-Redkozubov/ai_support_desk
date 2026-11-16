from app.prompts.support_chat.v1_0_0 import SUPPORT_CHAT_PROMPT_V1_0_0
from app.prompts.template import PromptTemplate, PromptTemplateError

ACTIVE_SUPPORT_CHAT_VERSION = "1.0.0"

_PROMPTS: dict[str, PromptTemplate] = {
    prompt.version: prompt for prompt in (SUPPORT_CHAT_PROMPT_V1_0_0,)
}


def get_support_chat_prompt(version: str | None = None) -> PromptTemplate:
    selected = ACTIVE_SUPPORT_CHAT_VERSION if version is None else version.strip()
    if selected not in _PROMPTS:
        raise PromptTemplateError(
            f"Неизвестная версия support_chat: {selected}. "
            f"Доступны: {', '.join(_PROMPTS)}."
        )
    return _PROMPTS[selected]