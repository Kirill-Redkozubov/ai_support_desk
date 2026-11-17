from app.config import Settings
from app.llm.client import LLMClient

client = LLMClient(Settings())
try:
    for event in client.stream([{'role': 'user', 'content': 'Посчитай словами от одного до двадцати.'}]):
        print(event)
finally:
    client.close()

