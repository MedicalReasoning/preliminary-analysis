from typing import TypeVar, Generic, Callable, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from runbox.benchmarks import BenchInput, BenchOutput, BenchEvalResult, SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


class SelfRefineAgent(
    Generic[BenchInput, BenchOutput, BenchEvalResult],
    SupportsBenchmark[BenchInput, BenchOutput, BenchEvalResult]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        main_prompt_path: str,
        critic_prompt_path: str,
        refiner_prompt_path: str,
        add_extractor: ExtractorAdder,
        n_iter: int = 1
    ) -> None:
        self.main = load_chat_prompt_template_json(main_prompt_path)\
            | ChatOpenAI(**main_config)
        self.critic = load_chat_prompt_template_json(critic_prompt_path)\
            | ChatOpenAI(**critic_config)
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)
        self.parser = add_extractor(self.parse)

        assert n_iter > 0
        self.n_iter = n_iter

    def run(self, input: BenchInput) -> dict:
        initial_response = invoke(self.main, input)
        output: dict = {
            "initial_response": initial_response,
            "initial_prediction": self.parser(initial_response),
            "iteration": []
        }

        for _ in range(self.n_iter):
            critic_response = invoke(self.critic, { **input, "initial_response": initial_response })
            refiner_response = invoke(self.refiner, { **input, "initial_response": initial_response, "critic_response": critic_response })

            output["iteration"].append({ # type: ignore[attr-defined]
                "critic_response": critic_response,
                "refiner_response": refiner_response,
                "refiner_prediction": self.parser(refiner_response)
            })

            initial_response = refiner_response

        return output

    def evaluate(
        self,
        evaluator: Callable[[BenchOutput, BenchOutput | None], BenchEvalResult],
        label: BenchOutput,
        output: dict
    ) -> list[BenchEvalResult]:
        predictions = [output["initial_prediction"]]
        for o in output["iteration"]:
            predictions.append(o["refiner_prediction"])
        return [evaluator(label, prediction) for prediction in predictions]