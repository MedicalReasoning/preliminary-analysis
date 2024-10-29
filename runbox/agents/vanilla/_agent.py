from typing import TypeVar, Generic, Callable, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from runbox.benchmarks import BenchInput, BenchOutput, BenchEvalResult, SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


class VanillaAgent(
    Generic[BenchInput, BenchOutput, BenchEvalResult],
    SupportsBenchmark[BenchInput, BenchOutput, BenchEvalResult]
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

    def run(self, input: BenchInput) -> dict:
        content = invoke(self.client, input)
        return { "prediction": self.parser(content), "output": content }

    def evaluate(
        self,
        evaluator: Callable[[BenchOutput, BenchOutput | None], BenchEvalResult],
        label: BenchOutput,
        output: dict
    ) -> BenchEvalResult:
        return evaluator(label, output["prediction"])