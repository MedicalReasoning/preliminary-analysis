import os
from typing import Callable, TypeVar, Mapping, Any
import json

from runbox.agents import *
from runbox.benchmarks import *
from runbox.benchmarks.benchmarks.medqa import MedQAEvalResult, MedQAInput, MedQAOutput
from runbox.utils import ChatOpenAIConfig, create_4o_mini_extractor
from runbox.agents.multi_critic_self_refine._agent import MultiCriticSelfRefineAgent


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")


type MultiCriticSelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]\
    = Callable[
        [ChatOpenAIConfig, ChatOpenAIConfig, ChatOpenAIConfig],
        MultiCriticSelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]
    ]

def mcsr_prompt_paths(benchmark: str) -> tuple[str, str, str, str]:
    return (
        f"runbox/prompts/{benchmark}/multi_critic_self_refine/main.json",
        f"runbox/prompts/{benchmark}/multi_critic_self_refine/critics",
        f"runbox/prompts/{benchmark}/multi_critic_self_refine/refiner.json",
        f"runbox/prompts/{benchmark}/extractor.json"
    )

def create_sr_agent(
    benchmark: str,
    AgentType: type[MultiCriticSelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]],
) -> MultiCriticSelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]: # type: ignore
    paths = mcsr_prompt_paths(benchmark)

    def f(
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig
    ) -> MultiCriticSelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]:
        return AgentType( # type: ignore
            main_config=main_config,
            critic_config=critic_config,
            refiner_config=refiner_config,
            main_prompt_path=paths[0],
            critic_prompts_dir_path=paths[1],
            refiner_prompt_path=paths[2],
            add_extractor=create_4o_mini_extractor(paths[3]),
            n_iter=2
        )

    return f

benchmark_configs: dict[str, tuple[type[Benchmark], MultiCriticSelfRefineAgentCreator]] = {
    "medqa": (MedQA, create_sr_agent("medqa", MedQAMultiCriticSelfRefineAgent)),
    "ddxplus": (DDXPlus, create_sr_agent("ddxplus", DDXPlusMultiCriticSelfRefineAgent))
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
    refiner: str
) -> tuple[type[Benchmark], MultiCriticSelfRefineAgent]:
    benchmark_, create_agent = benchmark_configs[benchmark]

    return (
        benchmark_,
        create_agent(
            model_configs[main],
            model_configs[critic],
            model_configs[refiner]
        )
    )