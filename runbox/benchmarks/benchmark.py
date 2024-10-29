from abc import ABC, abstractmethod
import os
from typing import Generic, TypeVar, Iterable, Iterator, Sized, Mapping, Any

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from tqdm import tqdm


def __validate_split(split: str) -> bool:
    return "+" not in split\
        and "[" not in split\
        and "]" not in split

def __validate_slice(slice: tuple[int, int]) -> bool:
    return 0 <= slice[0] <= 100\
        and 0 <= slice[1] <= 100\
        and slice[0] <= slice[1]

def __apply_slice(split: str, slice: tuple[int, int]) -> str:
    return split + f"[{slice[0]}%:{slice[1]}%]"

def _load_dataset(
    *args,
    split: str,
    slice: tuple[int, int] | None = None,
    **kwargs
) -> Dataset:
    assert __validate_split(split), "`split` must be without combining or slicing."

    if slice is not None:
        assert __validate_slice(slice), "Invalid `slice`. Are you sure the value is based on an integer percentage?"
        split = __apply_slice(split, slice)

    return load_dataset(
        *args,
        split=split,
        cache_dir=os.environ["HF_CACHE_DIR"],
        **kwargs
    )


BenchInput = TypeVar("BenchInput", bound=Mapping[str, Any])
BenchOutput = TypeVar("BenchOutput")
BenchEvalResult = TypeVar("BenchEvalResult")
_PreprocessedRow = tuple[BenchInput, BenchOutput]

class Benchmark(
    ABC,
    Generic[BenchInput, BenchOutput, BenchEvalResult],
    Sized,
    Iterable[_PreprocessedRow]
):
    def __init__(
        self,
        *args,
        split: str,
        slice: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        self._dataset = _load_dataset(*args, split=split, slice=slice, **kwargs)

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[_PreprocessedRow]:
        for row in tqdm(self._dataset):
            yield self.preprocess_row(row)

    @abstractmethod
    def preprocess_row(self, row: dict) -> _PreprocessedRow:
        ...

    @abstractmethod
    def evaluate_output(self, label: BenchOutput, prediction: BenchOutput | None) -> BenchEvalResult:
        ...