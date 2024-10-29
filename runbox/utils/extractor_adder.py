from typing import Callable, TypeVar
import re

from langchain_openai import ChatOpenAI

from .langchain import load_chat_prompt_template_json, invoke


_BenchOutput = TypeVar("_BenchOutput")
_Parser = Callable[[str], _BenchOutput | None]

ExtractorAdder = Callable[[_Parser], _Parser]

def _create_extractor_adder(extractor: Callable[[str], str | None]) -> ExtractorAdder:
    def add_extractor(parser: _Parser) -> _Parser:
        def extractor_added(output: str) -> _BenchOutput | None:
            if (extracted_str := extractor(output)) is not None:
                return parser(extracted_str)
            else:
                return None
        return extractor_added
    return add_extractor


_pattern = re.compile(r'```(.*?)```')
def _extract(output: str) -> str | None:
    matches = _pattern.findall(output, re.DOTALL)
    if len(matches) > 0:
        return matches[-1]
    else:
        return None

def create_4o_mini_extractor(prompt_path: str) -> ExtractorAdder:
    gpt_4o_mini = load_chat_prompt_template_json(prompt_path)\
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def extract_4o_mini(output: str) -> str | None:
        if (first_try := _extract(output)) is not None:
            return first_try
        else:
            second_try = invoke(gpt_4o_mini, { "output": output })[3:-3]
            if second_try in ["None", None, ""]:
                return None
            else:
                return second_try

    return _create_extractor_adder(extract_4o_mini)


add_extractor = _create_extractor_adder(_extract)