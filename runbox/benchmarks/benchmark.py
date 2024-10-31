from abc import ABC, abstractmethod
import os
from typing import TypeVar, Iterable, Iterator, Sized, Mapping, Any

from datasets import load_dataset # type: ignore
from datasets.arrow_dataset import Dataset # type: ignore
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
) -> Dataset: # type: ignore
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


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput", contravariant=True)
_BenchEvalResult = TypeVar("_BenchEvalResult")
_PreprocessedRow = tuple[_BenchInput, _BenchOutput]

from typing import Generic

class Benchmark(
    ABC,
    Generic[_BenchInput, _BenchOutput, _BenchEvalResult],
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
            yield self.preprocess_row(row) # type: ignore

    @abstractmethod
    def preprocess_row(self, row: dict) -> _PreprocessedRow:
        ...

    @abstractmethod
    def evaluate_output(self, label: _BenchOutput, prediction: _BenchOutput | None) -> _BenchEvalResult:
        ...