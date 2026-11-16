from app.prompts.support_summary.v1_0_0 import SUPPORT_SUMMARY_PROMPT_V1_0_0
from app.prompts.support_summary.v1_1_0 import SUPPORT_SUMMARY_PROMPT_V1_1_0
from app.prompts.template import PromptTemplate, PromptTemplateError

ACTIVE_SUPPORT_SUMMARY_VERSION = "1.1.0"

# «версия → промпт»: тот же словарь, что в скелете
_PROMPTS: dict[str, PromptTemplate] = {
    prompt.version: prompt
    for prompt in (SUPPORT_SUMMARY_PROMPT_V1_0_0, SUPPORT_SUMMARY_PROMPT_V1_1_0)
}


def get_support_summary_prompt(version: str | None = None) -> PromptTemplate:
    selected = ACTIVE_SUPPORT_SUMMARY_VERSION if version is None else version.strip()
    if selected not in _PROMPTS:
        raise PromptTemplateError(
            f"Неизвестная версия support_summary: {selected}. "
            f"Доступны: {', '.join(_PROMPTS)}."
        )
    return _PROMPTS[selected]