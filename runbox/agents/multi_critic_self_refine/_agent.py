from typing import Callable, TypeVar, Mapping, Any
import os
from pathlib import Path
from statistics import mean
from functools import reduce

from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type _SelfRefineRowResult = list[_BenchEvalResult]

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
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)
        self.parser = add_extractor(self.parse)
        self.n_iter = n_iter

    def _run_critic(self, input: _BenchInput, initial_response: str) -> tuple[list[str], str, float, list[int], bool]:
        responses = []
        scores = []
        total_cost = 0

        for critic in self.critics:
            critic_response, critic_cost = invoke(
                critic,
                { **input, "initial_response": initial_response } # type: ignore
            )

            feedback = "\n".join(filter(
                lambda x: '- GOOD:' in x or '- BAD:' in x,
                critic_response.splitlines()
            ))
            responses.append(feedback)

            s = self.parser(critic_response)
            score = int(s) if s is not None else 0
            scores.append(score)

            total_cost += critic_cost

        agg_response = "\n\n".join(responses)
        stop = reduce(lambda a, b: a and b, map(lambda x: x >= 4, scores))

        return responses, agg_response, total_cost, scores, stop

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
                critic_responses, critic_response, critic_cost, scores, stop\
                    = self._run_critic(input, initial_response)
            else:
                critic_responses = []
                critic_response = "-"
                critic_cost = 0
                scores = []

            if not stop:
                refiner_response, refiner_cost = invoke(
                    self.refiner,
                    { **input, "initial_response": initial_response, "critic_response": critic_response } # type: ignore
                )
                refiner_prediction = self.parser(refiner_response)
                output["iteration"].append({ # type: ignore
                    "critic_details": critic_responses,
                    "critic_response": critic_response,
                    "scores": scores,
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
                    "scores": scores,
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