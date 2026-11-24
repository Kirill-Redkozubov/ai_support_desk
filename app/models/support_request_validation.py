# проверка уже готового pydantic из json модели и реального запроса клиента
# например чтоб в pydantic был настоящий order_id клиента а не выдуманный
import re

from app.models.support_request import SupportRequestAnalysis


class SupportRequestSemanticError(ValueError):
    def __init__(self, issues: tuple[str, ...]) -> None: # заменяет init от ValueError
        self.issues = issues
        super().__init__(' '.join(issues))  # для вывода issues в принте

def validate_support_request(analysis: SupportRequestAnalysis, user_text: str) -> None:
    issues: list[str] = []
    # проверки для order_id, product, problem_description в missing_fields
    fields_to_check = (
        (
            'order_id',
            analysis.order_id,
        ),
        (
            'product',
            analysis.product,
        ),
        (
            'problem_description',
            analysis.problem_description,
        )
    )

    for field_name, value in fields_to_check:
        is_marked_missing = (
            field_name in analysis.missing_fields
        )

        if value is None and not is_marked_missing:
            issues.append(f'{field_name} равен null, но не указан в missing_fields')

        if value is not None and is_marked_missing:
            issues.append(f'{field_name} заполнен, но не указан в missing_fields.')

    # проверка что order_id из сообщения юзера не выдуман
    if analysis.order_id is not None:
        order_ids = set(re.findall(r"\b\d+\b", user_text))
        if analysis.order_id not in order_ids:
            issues.append(
                "order_id отсутствует в исходном обращении."
            )

    if issues:
        raise SupportRequestSemanticError(tuple(issues))

# пример работы проверки:

# user_text = "Кофемашина из заказа 8041 не включается."
#
# analysis = SupportRequestAnalysis(
#     category=SupportCategory.TECHNICAL_ISSUE,
#     order_id="9999",
#     product="кофемашина",
#     urgency=Urgency.NORMAL,
#     missing_fields=[],
# )
#
# validate_support_request(analysis, user_text)

