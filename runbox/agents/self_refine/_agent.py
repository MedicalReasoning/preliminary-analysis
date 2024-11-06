from typing import Callable, TypeVar, Mapping, Any

from langchain_openai import ChatOpenAI

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type _SelfRefineRowResult = list[_BenchEvalResult]

def _stop(response: str) -> bool:
    response_ = response.lower()
    return "`stop`" in response_\
        or "'stop'" in response_\
        or '"stop"' in response_

class SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, _SelfRefineRowResult]
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

    def run(self, input: _BenchInput) -> dict:
        initial_response, initial_cost = invoke(self.main, input)
        output: dict = {
            "initial_response": initial_response,
            "initial_prediction": self.parser(initial_response),
            "initial_cost": initial_cost,
            "iteration": []
        }

        stop = False
        prediction = output["initial_prediction"]
        for _ in range(self.n_iter):
            if not stop:
                critic_response, critic_cost = invoke(
                    self.critic,
                    { **input, "initial_response": initial_response } # type: ignore
                )
                stop = _stop(critic_response)
            else:
                critic_response = "-"
                critic_cost = 0

            if not stop:
                refiner_response, refiner_cost = invoke(
                    self.refiner,
                    { **input, "initial_response": initial_response, "critic_response": critic_response } # type: ignore
                )
                refiner_prediction = self.parser(refiner_response)
                output["iteration"].append({ # type: ignore
                    "critic_response": critic_response,
                    "refiner_response": refiner_response,
                    "refiner_prediction": refiner_prediction
                })
                initial_response = refiner_response
                prediction = refiner_prediction
            else:
                refiner_response = "-"
                refiner_cost = 0
                output["iteration"].append({ # type: ignore
                    "critic_response": critic_response,
                    "refiner_response": refiner_response,
                    "refiner_prediction": prediction,
                    "critic_cost": critic_cost,
                    "refiner_cost": refiner_cost
                })

        return output

    def evaluate(
        self,
        evaluator: Callable[[_BenchOutput, _BenchOutput | None], _BenchEvalResult],
        label: _BenchOutput,
        output: dict
    ) -> _SelfRefineRowResult:
        predictions = [output["initial_prediction"]]
        for o in output["iteration"]:
            predictions.append(o["refiner_prediction"])
        return [evaluator(label, prediction) for prediction in predictions]