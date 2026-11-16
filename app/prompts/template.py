from dataclasses import dataclass
from string import Template


class PromptTemplateError(ValueError): ...

@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str                                    # какая AI-функция
    version: str                                      # какой вариант её поведения
    body: str                                         # текст с переменными $name
    required_variables: frozenset[str]                # без них собрать нельзя
    optional_variables: frozenset[str] = frozenset()  # не передали → пустая строка

    def __post_init__(self) -> None:
        # Проверки при СОЗДАНИИ: ошибки в самом шаблоне видны сразу при запуске
        if not self.prompt_id.strip() or not self.version.strip():
            raise PromptTemplateError("prompt_id и version не должны быть пустыми.")
        if self.required_variables & self.optional_variables:
            raise PromptTemplateError(
                "Переменная не может быть сразу обязательной и необязательной."
            )
        template = Template(self.body)
        if not template.is_valid():
            raise PromptTemplateError("В тексте шаблона некорректная переменная.")
        found = set(template.get_identifiers())
        declared = self.required_variables | self.optional_variables
        if found != declared:
            raise PromptTemplateError(
                f"Переменные в тексте {sorted(found)} "
                f"не совпадают с объявленными {sorted(declared)}."
            )

    def render(self, **variables: str) -> str:
        # Проверки при СБОРКЕ: правильные ли значения передали
        provided = set(variables)
        missing = self.required_variables - provided
        if missing:
            raise PromptTemplateError(f"Не переданы: {', '.join(sorted(missing))}.")
        unknown = provided - self.required_variables - self.optional_variables
        if unknown:
            raise PromptTemplateError(f"Неизвестные переменные: {', '.join(sorted(unknown))}.")
        if not all(isinstance(value, str) for value in variables.values()):
            raise PromptTemplateError("Значения переменных должны быть строками.")

        values = {name: "" for name in self.optional_variables}  # по умолчанию пустые
        values.update(variables)
        return Template(self.body).substitute(values).strip()