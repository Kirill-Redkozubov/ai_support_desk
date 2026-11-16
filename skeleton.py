# Скелет кароче проекта без брони

from openai import OpenAI
from app.config import Settings

from string import Template

settings = Settings()

client = OpenAI(
    base_url=str(settings.base_url),
    api_key=settings.llm_api_key.get_secret_value(),
)

PROMPTS = {
    "1.0.0": (
        "Ты отвечаешь клиенту в чате службы поддержки. "
        "Опирайся только на сведения из этого диалога. "
        "Не придумывай данные о заказе и не обещай возврат, замену или сроки. "
        "Если данных не хватает, задай один уточняющий вопрос. Отвечай кратко."
    ),
}
instruction = PROMPTS["1.0.0"]

history = []

while True:
    user_text = input('\nВы: ')
    if user_text.strip() == '/exit':
        break

    user_message = {
        "role": "user",
        "content": f"<customer_message>\n{user_text}\n</customer_message>",
    }

    messages = [
        {'role': 'developer', 'content': instruction},
        *history,
        user_message,
    ]

    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        max_completion_tokens=settings.max_output_tokens,
    )
    answer = response.choices[0].message.content
    print(f'Assistant: {answer}')

    history += [user_message, {'role': 'assistant', 'content': answer}]
    print(f'Сообщений в истории: {len(history)}, '
          f'входных токенов: {response.usage.prompt_tokens}')