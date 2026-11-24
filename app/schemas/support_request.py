from app.models.support_request import SupportRequestAnalysis

SUPPORT_REQUEST_SCHEMA_NAME = "support_request"

# класс Pydantic генерирует готовую json_schema для openai модели
SUPPORT_REQUEST_SCHEMA = SupportRequestAnalysis.model_json_schema()