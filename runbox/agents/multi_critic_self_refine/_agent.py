from typing import Callable, TypeVar, Mapping, Any
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type _SelfRefineRowResult = list[_BenchEvalResult]

def _stop(response: str) -> bool:
    response_ = response.lower()
    return "`yes`" in response_\
        or "'yes'" in response_\
        or '"yes"' in response_

class MultiCriticSelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, _SelfRefineRowResult]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        main_prompt_path: str,
        critic_prompts_dir_path: str,
        agg_critic_prompt_path: str,
        refiner_prompt_path: str,
        add_extractor: ExtractorAdder,
        n_iter: int = 3
    ) -> None:
        self.main = load_chat_prompt_template_json(main_prompt_path)\
            | ChatOpenAI(**main_config)
        self.critics = [
            load_chat_prompt_template_json(critic_prompts_dir_path / Path(prompt_path))\
                | ChatOpenAI(**critic_config)
            for prompt_path in os.listdir(critic_prompts_dir_path)
        ]
        self.agg_critic = load_chat_prompt_template_json(agg_critic_prompt_path)\
            | ChatOpenAI(**critic_config)
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)
        self.parser = add_extractor(self.parse)
        self.n_iter = n_iter

    def _run_critic(self, input: _BenchInput, initial_response: str) -> tuple[list[str], str, float, bool]:
        responses = []
        total_cost = 0

        for critic in self.critics:
            critic_response, critic_cost = invoke(
                critic,
                { **input, "initial_response": initial_response } # type: ignore
            )
            responses.append(critic_response)
            total_cost += critic_cost

        agg_response, agg_cost = invoke(
            self.agg_critic,
            {
                **input,
                "initial_response": initial_response,
                **{ f"critic_{i}": res for i, res in enumerate(responses) }
            }
        )
        total_cost += agg_cost
        stop = _stop(agg_response)

        return responses, agg_response, total_cost, stop

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
                critic_responses, critic_response, critic_cost, stop\
                    = self._run_critic(input, initial_response)
            else:
                critic_responses = []
                critic_response = "-"
                critic_cost = 0

            if not stop:
                refiner_response, refiner_cost = invoke(
                    self.refiner,
                    { **input, "initial_response": initial_response, "critic_response": critic_response } # type: ignore
                )
                refiner_prediction = self.parser(refiner_response)
                output["iteration"].append({ # type: ignore
                    "critic_details": critic_responses,
                    "critic_response": critic_response,
                    "refiner_response": refiner_response,
                    "refiner_prediction": refiner_prediction,
                    "critic_cost": critic_cost,
                    "refiner_cost": refiner_cost
                })
                initial_response = refiner_response
                prediction = refiner_prediction
            else:
                refiner_response = "-"
                refiner_cost = 0
                output["iteration"].append({ # type: ignore
                    "critic_details": critic_responses,
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