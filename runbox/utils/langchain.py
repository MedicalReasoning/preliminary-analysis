from typing import TypedDict, Mapping, Any, cast
import json

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


ChatOpenAIConfig = dict


def load_chat_prompt_template_json(json_path: str) -> ChatPromptTemplate:
    messages = [*map(tuple, json.load(open(json_path, "r")))] # type: ignore
    return ChatPromptTemplate.from_messages(messages)


def invoke(client: Runnable, params: Mapping[str, Any]) -> str:
    content: str = client.invoke(cast(dict[str, str], params)).content # type: ignore[assignment]
    return content