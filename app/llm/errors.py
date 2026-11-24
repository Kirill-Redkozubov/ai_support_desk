class LLMClientError(RuntimeError):
    """Общий предок всех ошибок нашего LLM-клиента."""
    # супер типо наследует все от RuntimeError
    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id  # ID запроса у провайдера — для логов/поддержки
    # вот тут хз
    def __str__(self) -> str:
        text = super().__str__()
        if self.request_id is None:
            return text
        return f"{text} ID запроса: {self.request_id}"

# ... значит pass только наследуемое
class LLMAccessError(LLMClientError): ...  # 401/403 — ключ или права


class LLMBalanceError(LLMClientError): ...  # 402 — нет денег на балансе


class LLMRequestError(LLMClientError): ...  # 400/404/422 — модель или параметры


class LLMRateLimitError(LLMClientError): ...  # 429 — слишком часто


class LLMTimeoutError(LLMClientError): ...  # не дождались ответа


class LLMConnectionError(LLMClientError): ...  # нет связи


class LLMProviderError(LLMClientError): ...  # 5xx или странный ответ
