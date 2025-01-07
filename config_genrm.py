import os
from typing import Callable, TypeVar, Mapping, Any
import json

from runbox.agents import *
from runbox.benchmarks import *
from runbox.benchmarks.benchmarks.medqa import MedQAEvalResult, MedQAInput, MedQAOutput
from runbox.utils import ChatOpenAIConfig, create_4o_mini_extractor
from runbox.agents.genrm._agent import GenRMAgent


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")


type GenRMAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]\
    = Callable[
        [ChatOpenAIConfig, ChatOpenAIConfig, ChatOpenAIConfig, ChatOpenAIConfig],
        GenRMAgent[_BenchInput, _BenchOutput, _BenchEvalResult]
    ]

def prompt_paths(benchmark: str) -> tuple[str, str, str, str, str]:
    return (
        f"runbox/prompts/{benchmark}/genrm/main.json",
        f"runbox/prompts/{benchmark}/genrm/critics",
        f"runbox/prompts/{benchmark}/genrm/agg_critic.json",
        f"runbox/prompts/{benchmark}/genrm/refiner.json",
        f"runbox/prompts/{benchmark}/extractor.json"
    )

def create_agent(
    benchmark: str,
    AgentType: type[GenRMAgent[_BenchInput, _BenchOutput, _BenchEvalResult]],
) -> GenRMAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]: # type: ignore
    paths = prompt_paths(benchmark)

    def f(
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        agg_critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig
    ) -> GenRMAgent[_BenchInput, _BenchOutput, _BenchEvalResult]:
        return AgentType( # type: ignore
            main_config=main_config,
            critic_config=critic_config,
            agg_critic_config=agg_critic_config,
            refiner_config=refiner_config,
            main_prompt_path=paths[0],
            critic_prompts_dir_path=paths[1],
            agg_critic_prompt_path=paths[2],
            refiner_prompt_path=paths[3],
            add_extractor=create_4o_mini_extractor(paths[4]),
            n_iter=2
        )

    return f

benchmark_configs: dict[str, tuple[type[Benchmark], GenRMAgentCreator]] = {
    "medqa": (MedQA, create_agent("medqa", MedQAGenRMAgent)),
    "ddxplus": (DDXPlus, create_agent("ddxplus", DDXPlusGenRMAgent))
}


MODEL_CONFIGS_PATH = "models.json"
try:
    model_configs: dict = json.load(open(MODEL_CONFIGS_PATH, "r"))
except:
    raise Exception("model config file required")
def prepare(
    benchmark: str,
    main: str,
    critic: str,
    # agg_critic: str,
    refiner: str
) -> tuple[type[Benchmark], GenRMAgent]:
    benchmark_, create_agent = benchmark_configs[benchmark]

    return (
        benchmark_,
        create_agent(
            model_configs[main],
            model_configs[critic],
            model_configs['gpt-4o-mini'],
            model_configs[refiner]
        )
    )