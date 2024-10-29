from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any

from .benchmark import BenchInput, BenchOutput, BenchEvalResult


class SupportsBenchmark(ABC, Generic[BenchInput, BenchOutput, BenchEvalResult]):
    @abstractmethod
    def run(self, input: BenchInput) -> dict:
        ...

    @abstractmethod
    def parse(self, extracted_str: str) -> BenchOutput | None:
        ...

    @abstractmethod
    def evaluate(
        self,
        evaluator: Callable[[BenchOutput, BenchOutput | None], BenchEvalResult],
        label: BenchOutput,
        output: dict
    ) -> Any:
        ...