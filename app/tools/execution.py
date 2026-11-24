from datetime import UTC, datetime
from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FutureTimeoutError)
from dataclasses import dataclass
from enum import StrEnum
import json
from time import perf_counter

from pydantic import BaseModel, ValidationError

from app.llm.client import ToolCall
from app.tools.registry import ToolNotRegisteredError, ToolRegistry


class ToolBusinessError(RuntimeError):
    def __init__(self, message: str, code: str = 'business_error') -> None:
        super().__init__(message)
        self.code = code

class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, audit: 'ToolAuditRecord') -> None:
        super().__init__(message)
        self.audit = audit

class ToolNotAllowedError(ToolExecutionError): ...

class ToolArgumentsError(ToolExecutionError): ...

class ToolTimeoutError(ToolExecutionError): ...

class ToolTechnicalError(ToolExecutionError): ...


class ToolExecutionStatus(StrEnum):
    SUCCESS = 'success'
    BUSINESS_ERROR = 'business_error'
    REJECTED = 'rejected'
    TIMEOUT = 'timeout'
    TECHNICAL_ERROR = 'technical_error'

@dataclass(frozen=True)
class ToolAuditRecord:
    call_id: str
    tool_name: str
    started_at: str
    duration_seconds: float
    status: ToolExecutionStatus
    error_code: str | None

@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    audit: ToolAuditRecord


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError('Количество workers должно быть больше нуля.')

        self._registry = registry
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix='tool_executor')

# возвращает результат выбранной функции с audit в json str формате
    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()

        # замыкание чтобы не дублировать код Перминова 3.6 урок
        def audit(
                status: ToolExecutionStatus,
                error_code: str | None = None,
        ) -> ToolAuditRecord:
            return ToolAuditRecord(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                started_at=started_at,
                duration_seconds=perf_counter() - started,
                status=status,
                error_code=error_code,
            )

        # тут целая портянка одних только обработчиков ошибок
        try: # ищем готовый эк RegisteredTool в ToolsRegistry
            tool = self._registry.get(tool_call.name)
        except ToolNotRegisteredError as error: # эк ToolAuditRecord
            audit = audit(status=ToolExecutionStatus.REJECTED, error_code='unknown_tool')
            raise ToolNotAllowedError('Модель запросила неизвестный '
                                      'инструмент.', audit) from error

        try: # готовым полем эк RegisteredTool.arguments_model проверяем json строку
            # и получаем эк GetOrderStatusArguments например, или GetReturnPolicyArguments
            # у GetOrderStatusArguments.order_id поле, а у GetReturnPolicyArguments.product
            arguments = tool.arguments_model.model_validate_json(tool_call.arguments)
        except ValidationError as error:
            audit = audit(status=ToolExecutionStatus.REJECTED, error_code='invalid_arguments')
            raise ToolArgumentsError('Аргументы инструмента не прошли проверку.', audit) from error

        future = self._pool.submit(tool.handler, arguments)

        # result итоговый эк результата функции с полями
        try: # для ограничения времени на выполнение функции, например 2.0 сек
            result = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel() # если за 2.0 сек не успело функция не выполнится
            audit = audit(status=ToolExecutionStatus.TIMEOUT, error_code='timeout')
            raise ToolTimeoutError('Инструмент не завершился вовремя.', audit) from error
        except ToolBusinessError as error:
            audit = audit(status=ToolExecutionStatus.BUSINESS_ERROR, error_code=error.code)
            return ToolExecutionResult(
                content=json.dumps(
                    {
                        'ok': False,
                        'error': {
                            'code': error.code,
                            'message': str(error),
                        }
                    },
                    ensure_ascii=False,
                ),
                audit=audit,
            )
        except Exception as error:
            audit = audit(status=ToolExecutionStatus.TECHNICAL_ERROR, error_code='technical_error')
            raise ToolTechnicalError('Инструмент завершился технической ошибкой.', audit) from error

        if not isinstance(result, BaseModel):
            audit = audit(status=ToolExecutionStatus.TECHNICAL_ERROR, error_code='invalid_result_type')
            raise ToolTechnicalError('Инструмент вернул результат неверного типа.', audit)

        # тут наконец то успех
        audit = audit(status=ToolExecutionStatus.SUCCESS)
        return ToolExecutionResult(
            content=json.dumps( # словарь в json str
                {
                    'ok': True, # result obj парсим в dict но готовый для json str
                    'data': result.model_dump(mode='json')
                },
                ensure_ascii=False, # чтоб кирилица не превращалась в краказябры
            ),
            audit=audit,
        )

    # закрываем пул потоков всех
    def close(self) -> None: # подождать начавшиеся потоки и закрыться не начав потоки в очереди
        self._pool.shutdown(wait=True, cancel_futures=True)



























