import os
from typing import Callable, TypeVar, Mapping, Any

from runbox.agents import *
from runbox.benchmarks import *
from runbox.benchmarks.benchmarks.medqa import MedQAEvalResult, MedQAInput, MedQAOutput
from runbox.utils import ChatOpenAIConfig, create_4o_mini_extractor
from runbox.agents.self_refine._agent import SelfRefineAgent


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")


type SelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]\
    = Callable[
        [ChatOpenAIConfig, ChatOpenAIConfig, ChatOpenAIConfig],
        SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]
    ]

def sr_prompt_paths(benchmark: str) -> tuple[str, str, str, str]:
    return (
        f"runbox/prompts/{benchmark}/self_refine/main.json",
        f"runbox/prompts/{benchmark}/self_refine/critic.json",
        f"runbox/prompts/{benchmark}/self_refine/refiner.json",
        f"runbox/prompts/{benchmark}/extractor.json"
    )

def create_sr_agent(
    benchmark: str,
    AgentType: type[SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]],
) -> SelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]: # type: ignore
    paths = sr_prompt_paths(benchmark)

    def f(
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig
    ) -> SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]:
        return AgentType( # type: ignore
            main_config=main_config,
            critic_config=critic_config,
            refiner_config=refiner_config,
            main_prompt_path=paths[0],
            critic_prompt_path=paths[1],
            refiner_prompt_path=paths[2],
            add_extractor=create_4o_mini_extractor(paths[3]),
            n_iter=3
        )

    return f

benchmark_configs: dict[str, tuple[type[Benchmark], SelfRefineAgentCreator]] = {
    "medqa": (MedQA, create_sr_agent("medqa", MedQASelfRefineAgent)),
    "medmcqa": (MedMCQA, create_sr_agent("medmcqa", MedMCQASelfRefineAgent)),
    "pubmedqa": (PubMedQA, create_sr_agent("pubmedqa", PubMedQASelfRefineAgent)),
    # "ddxplus": (DDXPlus, create_sr_agent("ddxplus", DDXPlusSelfRefineAgent))
}

def model_configs(model: str) -> ChatOpenAIConfig:
    return {
        "4o": {
            "model": "gpt-4o",
            "temperature": 0
        },
        "4o_mini": {
            "model": "gpt-4o-mini",
            "temperature": 0
        },
        "meerkat_8b": {
            "model": "dmis-lab/llama-3-meerkat-8b-v1.0",
            "temperature": 0,
            "base_url": os.environ["MEERKAT_8B_BASE_URL"],
            "stop": "<|eot_id|>"
        },
        "meditron_70b": {
            "model": "epfl-llm/meditron-70b",
            "temperature": 0,
            "base_url": os.environ["MEDITRON_70B_BASE_URL"]
        },
        "llama_8b": {
            "model": "meta-llama/Llama-3.1-8B",
            "temperature": 0,
            "base_url": os.environ["LLAMA_8B_BASE_URL"]
        },
    }[model]

def prepare(
    benchmark: str,
    main: str,
    critic: str,
    refiner: str
) -> tuple[type[Benchmark], SelfRefineAgent]:
    benchmark_, create_agent = benchmark_configs[benchmark]

    return (
        benchmark_,
        create_agent(
            model_configs(main),
            model_configs(critic),
            model_configs(refiner)
        )
    )