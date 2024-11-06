from typing import TypedDict, Mapping, Any, cast, TypeVar, ParamSpec, Callable
import json
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback # type: ignore


ChatOpenAIConfig = dict


def load_chat_prompt_template_json(json_path: str) -> ChatPromptTemplate:
    messages = [*map(tuple, json.load(open(json_path, "r")))] # type: ignore
    return ChatPromptTemplate.from_messages(messages)


_P = ParamSpec("_P")
_T = TypeVar("_T")

def track_cost(f: Callable[_P, _T]) -> Callable[_P, tuple[_T, float]]:
    @wraps(f)
    def f_tracking(*args: _P.args, **kwargs: _P.kwargs) -> tuple[_T, float]:
        with get_openai_callback() as cb:
            result = f(*args, **kwargs)
            return (result, cb.total_cost)
    return f_tracking


@track_cost
def invoke(client: Runnable, params: Mapping[str, Any]) -> str:
    content: str = client.invoke(cast(dict[str, str], params)).content # type: ignore[assignment]
    return content