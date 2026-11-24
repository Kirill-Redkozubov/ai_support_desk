"""Эксперимент: Responses API + история на сервере + compaction через ProxyAPI.

Не часть проекта. Запуск: python try_responses.py
"""

from openai import OpenAI

from app.config import Settings

settings = Settings()
client = OpenAI(base_url=str(settings.base_url), api_key=settings.llm_api_key.get_secret_value())

INSTRUCTION = "Ты отвечаешь клиенту поддержки интернет-магазина. Отвечай кратко, 1-2 предложения."
THRESHOLD = 1500  # порог сжатия в токенах: маленький, чтобы сработал за пару ходов
FILLER = "Ещё подробности: " + "курьер опоздал на два часа, коробка мятая, скотч переклеен. " * 30

TURNS = [
    "Здравствуйте. Номер моего заказа 7718, монитор пришёл с треснутой подставкой.",
    FILLER,
    FILLER,
    FILLER,
    "Напомните, какой номер заказа я назвал в самом начале?",
]

previous_id = None
for number, text in enumerate(TURNS, start=1):
    response = client.responses.create(
        model=settings.model,
        instructions=INSTRUCTION,          # с previous_response_id инструкция не наследуется, шлём каждый раз
        input=text,                        # только новое сообщение, без истории
        previous_response_id=previous_id,  # историю подставляет сервер
        max_output_tokens=1000,
        # параметр новый, поэтому передаём «сырым» полем JSON, мимо проверок SDK
        extra_body={"context_management": [{"type": "compaction", "compact_threshold": THRESHOLD}]},
    )
    previous_id = response.id
    item_types = [item.type for item in response.output]
    print(f"Ход {number}: вход={response.usage.input_tokens} выход={response.usage.output_tokens} items={item_types}")
    print(f"  Ответ: {response.output_text}")
    print(f"  status={response.status} incomplete={response.incomplete_details} error={response.error}")
