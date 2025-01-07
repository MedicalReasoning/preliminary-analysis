from typing import Callable, TypeVar, Mapping, Any
import os
from pathlib import Path
from statistics import mean
from functools import reduce

from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
import numpy as np

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder, track_cost


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type _GenRMRowResult = list[_BenchEvalResult]


INF = float('-inf')
def _calulate_prob(top_logprobs: list[dict]) -> tuple[float, float]:
    positive_tokens = ['Yes', 'YES', 'yes']
    negative_tokens = ['No', 'NO', 'no']

    positive_logprob = INF
    negative_logprob = INF

    for item in top_logprobs:
        token = item['token']
        logprob = item['logprob']
        if token in positive_tokens:
            positive_logprob = np.logaddexp(positive_logprob, logprob)
        elif token in negative_tokens:
            negative_logprob = np.logaddexp(negative_logprob, logprob)

    if (positive_logprob == INF) and (negative_logprob == INF):
        return 0.5, 0.5
    if positive_logprob == INF:
        return 0.0, 1.0
    if negative_logprob == INF:
        return 1.0, 0.0
    logits = np.array([positive_logprob, negative_logprob])
    probs = np.exp(logits) / np.sum(np.exp(logits))

    positive_prob = probs[0]
    negative_prob = probs[1]

    return positive_prob, negative_prob

def _calculate_neg(response: ChatGeneration) -> float | None:
    lever = False
    for step in response.generation_info['logprobs']['content']: # type: ignore
        match (step['token'] == '```', lever):
            case (True, False):
                lever = True
            case (False, True):
                pos, neg = _calulate_prob(step['top_logprobs'])
                return neg
            case (True, True):
                break
            case _:
                continue
    return None

@track_cost
def _run_single_critic(
    model: ChatOpenAI,
    prompt: ChatPromptTemplate,
    params: dict,
    aggregator: Runnable
) -> tuple[str, float]:
    input = prompt.invoke(params)
    responses = model._generate(input) # type: ignore

    feedback_lines = sum(
        [[*filter(
            lambda x: '- GOOD:' in x or '- BAD:' in x,
            gen.text.splitlines()
        )] for gen in responses.generations],
        []
    )
    feedback = aggregator.invoke({
        **params,
        "feedback": "\n".join(feedback_lines)
    }).content # type: ignore

    negs = [*filter(
        lambda x: x is not None,
        [*map(_calculate_neg, responses.generations)]
    )]
    neg = mean(negs) if len(negs) > 0 else 0 # type: ignore

    return feedback, neg


class GenRMAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, _GenRMRowResult]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        agg_critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        main_prompt_path: str,
        critic_prompts_dir_path: str,
        agg_critic_prompt_path: str,
        refiner_prompt_path: str,
        add_extractor: ExtractorAdder,
        n_iter: int = 3,
        critic_n: int = 1,
        critic_temp: float = 1.,
        critic_top_logprobs: int = 20
    ) -> None:
        self.main = load_chat_prompt_template_json(main_prompt_path)\
            | ChatOpenAI(**main_config)
        self.critic = ChatOpenAI(
            **critic_config,
            n=critic_n,
            temperature=critic_temp,
            logprobs=True,
            top_logprobs=critic_top_logprobs
        )
        self.critic_prompts = [
            load_chat_prompt_template_json(critic_prompts_dir_path / Path(prompt_path))
            for prompt_path in os.listdir(critic_prompts_dir_path)
        ]
        self.agg_critic = load_chat_prompt_template_json(agg_critic_prompt_path)\
            | ChatOpenAI(**agg_critic_config)
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)
        self.parser = add_extractor(self.parse)
        self.n_iter = n_iter

    def _run_critic(self, input: _BenchInput, initial_response: str) -> tuple[list[str], str, float, list[float], bool]:
        responses: list[str] = []
        neg_scores: list[float] = []
        total_cost = 0

        for critic_prompt in self.critic_prompts:
            response, critic_cost = _run_single_critic(
                self.critic,
                critic_prompt,
                { **input, "initial_response": initial_response },
                self.agg_critic
            )
            feedback, neg_score = response

            responses.append(feedback)
            neg_scores.append(neg_score)
            total_cost += critic_cost

        agg_response = "\n\n".join(responses)

        stop = reduce(lambda a, b: a and b, map(lambda x: x >= 0.5, neg_scores))

        return responses, agg_response, total_cost, neg_scores, stop

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
                critic_responses, critic_response, critic_cost, neg_scores, stop\
                    = self._run_critic(input, initial_response)
            else:
                critic_responses = []
                critic_response = "-"
                critic_cost = 0
                neg_scores = []

            if not stop:
                refiner_response, refiner_cost = invoke(
                    self.refiner,
                    { **input, "initial_response": initial_response, "critic_response": critic_response } # type: ignore
                )
                refiner_prediction = self.parser(refiner_response)
                output["iteration"].append({ # type: ignore
                    "critic_details": critic_responses,
                    # "critic_response": critic_response,
                    "neg_scores": neg_scores,
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
                    # "critic_response": critic_response,
                    "neg_scores": neg_scores,
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
    ) -> _GenRMRowResult:
        predictions = [output["initial_prediction"]]
        for o in output["iteration"]:
            predictions.append(o["refiner_prediction"])
        return [evaluator(label, prediction) for prediction in predictions]