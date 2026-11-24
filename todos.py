# import json
# import re
#
# def execute_order_status_call(tool_call: dict, statuses: dict[str, str]) -> dict:
#     # Напишите реализацию.
#     if tool_call['name'] != 'get_order_status':
#         raise ValueError
#     try:
#         arguments = json.loads(tool_call['arguments'])
#         if type(arguments) != dict:
#             raise
#     except:
#         raise ValueError
#
#     order_id = arguments.get('order_id')
#
#     if len(arguments) != 1 or not order_id or type(order_id) != str or len(order_id) != 4 or not order_id.isdigit():
#         raise ValueError
#
#     status = statuses.get(order_id)
#     data = json.dumps({
#         'order_id': order_id,
#         'found': bool(status),
#         'status': status,
#     })
#     return {
#         'role': 'tool',
#         'tool_call_id': tool_call.id,
#         'content': data
#     }

# import json
# import re
#
# def answer_with_order_tool(client, user_text, tool_definition, statuses) -> str:
#     # Напишите реализацию.
#     if not user_text or type(user_text) != str or not user_text.strip():
#         raise ValueError
#     user_text = user_text.strip()
#     messages = [
#         {'role': 'user', 'content': user_text}
#     ]
#     response = client.generate_with_tools(
#         messages=messages,
#         tools=[tool_definition],
#         tool_choice='auto',
#     )
#     if response.finish_reason == 'stop':
#         if response.tool_calls or not response.text.strip():
#             raise ValueError
#         return response.text.strip()
#     if response.finish_reason == 'tool_calls':
#         if len(response.tool_calls) != 1:
#             raise ValueError
#
#     tool_call = response.tool_calls[0]
#     if tool_call.name != 'get_order_status':
#         raise ValueError
#     try:
#         arguments = json.loads(tool_call.arguments)
#         order_id = arguments.get('order_id')
#         if len(arguments) != 1 or not order_id or type(order_id) != str or len(order_id) != 4 or not order_id.isdigit():
#             raise
#     except:
#         raise ValueError
#
#     status = statuses.get('order_id')
#     data = json.dumps(
#         {
#             'order_id': order_id,
#             'found': bool(status),
#             'status': status,
#         }
#     )
#
#     messages.extend(
#         [
#             {
#                 'role': 'assistant',
#                 'content': response.text,
#                 'tool_calls': [
#                     {
#                         'id': tool_call.id,
#                         'type': 'function',
#                         'function': {
#                             'name': tool_call.name,
#                             'arguments': json.dumps(arguments)
#                         }
#                     }
#                 ]
#             },
#             {
#                 'role': 'tool',
#                 'tool_call_id': tool_call.id,
#                 'content': data
#             }
#         ]
#     )
#     response = client.generate_with_tools(
#         messages=messages,
#         tools=[tool_definition],
#         tool_choice='none'
#     )
#     text = response.text.strip()
#     if response.finish_reason != 'stop' or not text or response.tool_calls:
#         raise ValueError
#     return text


# class ToolRegistry:
#     def __init__(self) -> None:
#         # Создайте хранилище.
#         self._tools: dict = {}
#
#     def register(self, tool: dict) -> None:
#         if type(tool) != dict:
#             raise ValueError
#         name, definition, handler = (tool.get(elem) for elem in ('name', 'definition', 'handler'))
#         try:
#             check1, check2, check3 = bool(name.strip()), type(definition) == dict, callable(handler)
#             if not all((name, definition, handler)):
#                 raise
#         except:
#             raise ValueError
#
#         if self._tools.get(name):
#             raise ValueError
#         self._tools[name.strip()] = tool
#
#     def get(self, name: str) -> dict:
#         return self._tools.get(name)
#
#     def api_schemas(self) -> list[dict]:
#         result: list[dict] = []
#         return [elem.get('definition') for elem in self._tools.values()]
#
# import json
#
# def execute_registered_tool(registry, tool_call: dict) -> dict:
#     # Напишите реализацию.
#     try:
#         registry = registry.get(tool_call['name'])
#     except:
#         raise ValueError
#     arguments = json.loads(tool_call['arguments'])
#     if type(arguments) != dict:
#         raise ValueError
#
#     try:
#         normalized_arguments = registry['validator'](arguments)
#     except:
#         raise ValueError
#     try:
#         result = registry['handler'](normalized_arguments)
#     except:
#         raise RuntimeError('Инструмент не выполнился.')
#     return {
#         'tool_call_id': tool_call['id'],
#         'tool_name': tool_call['name'],
#         'result': result,
#         'status': 'success'
#     }

import json

def run_bounded_tool_loop(client, tools, executor, messages, max_steps):
    # Напишите реализацию.
    if type(max_steps) != int or max_steps <= 0:
        raise ValueError

    conversation = list(messages)
    audits = []
    steps = 0

    for step in range(1, max_steps + 1):
        steps += 1
        response = client.generate_with_tools(messages=conversation, tools=tools, tool_choice='auto')
        if response.finish_reason == 'stop':
            if type(response.text) != str or not response.text.strip():
                raise ValueError
            text = response.text.strip()
            if response.tool_calls or not text:
                raise ValueError
            return (text, steps, tuple(audits))

        if response.finish_reason == 'tool_calls':
            if len(response.tool_calls) != 1:
                raise ValueError
            tool_call = response.tool_calls[0]
            tool_result, audit = executor(tool_call)
            conversation.append({
                'role': 'assistant',
                'content': response.text,
                'tool_calls': [
                    {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.name,
                            'arguments': tool_call.arguments,
                        }
                    }
                ]
            })
            conversation.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': json.dumps(tool_result),
            })
            audits.append(audit)
        if response.finish_reason != 'tool_calls':
            raise ValueError

    raise ValueError






































