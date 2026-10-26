


from typing import Dict, List
from langchain_core.messages import convert_to_openai_messages, messages_to_dict

from api.models.kgraph_schemas import MessageDto

def msgdto_to_dict(message: MessageDto):
    return { "role":message.role.lower(), "content" : message.message }

def chatdtos_to_dict(messages: List[MessageDto]) -> List[Dict[str,str]]:
    return [msgdto_to_dict(msg) for msg in messages]

def dict_to_msgdtos(msgs_dict: List[Dict[str,str]]) -> List[MessageDto]:
    messages: List[MessageDto] = []
    for message in msgs_dict:
        keys = list(message.keys())
        messages.append(MessageDto(role=message[keys[0]], message=message[keys[1]]))
    return messages

def to_openai_messages(messages):
    return convert_to_openai_messages(messages)

def langchain_to_dict(messages):
    return messages_to_dict(messages)