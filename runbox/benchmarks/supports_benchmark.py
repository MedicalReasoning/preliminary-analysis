from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Mapping, Any, Generic


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

_AgentRowResult = TypeVar("_AgentRowResult", covariant=True)
type _AgentRowResults = list[_AgentRowResult]

class SupportsBenchmark(ABC, Generic[_BenchInput, _BenchOutput, _BenchEvalResult, _AgentRowResult]):
    @abstractmethod
    def run(self, input: _BenchInput) -> dict:
        ...

    @abstractmethod
    def parse(self, extracted_str: str) -> _BenchOutput | None:
        ...

    @abstractmethod
    def evaluate(
        self,
        evaluator: Callable[[_BenchOutput, _BenchOutput | None], _BenchEvalResult],
        label: _BenchOutput,
        output: dict
    ) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def calc_full_score(full: _AgentRowResults) -> Any:
        ...
