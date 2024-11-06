from typing import Callable, TypeVar, Mapping, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type VanillaRowResult = _BenchEvalResult

class VanillaAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, VanillaRowResult]
):
    def __init__(
        self,
        client_config: ChatOpenAIConfig,
        prompt_path: str,
        add_extractor: ExtractorAdder
    ) -> None:
        self.client = load_chat_prompt_template_json(prompt_path)\
            | ChatOpenAI(**client_config)
        self.parser = add_extractor(self.parse)

    def run(self, input: _BenchInput) -> dict:
        content, cost = invoke(self.client, input)
        return { "prediction": self.parser(content), "output": content, "cost": cost }

    def evaluate(
        self,
        evaluator: Callable[[_BenchOutput, _BenchOutput | None], _BenchEvalResult],
        label: _BenchOutput,
        output: dict
    ) -> _BenchEvalResult:
        return evaluator(label, output["prediction"])