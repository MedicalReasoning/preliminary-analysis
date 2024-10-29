from typing import TypedDict

from ..benchmark import Benchmark
from ..supports_benchmark import SupportsBenchmark


class MedMCQAInput(TypedDict):
    question: str
    options: str

MedMCQAOutput = str

MedMCQAEvalResult = bool

class MedMCQA(Benchmark[MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult]):
    def __init__(
        self,
        split: str,
        slice: tuple[int, int] | None = None
    ) -> None:
        assert split in ["train", "validation", "test"], "invalid `split`."
        super().__init__(
            "openlifescienceai/medmcqa",
            split=split,
            slice=slice
        )

    def preprocess_row(self, row: dict) -> tuple[MedMCQAInput, MedMCQAOutput]:
        input: MedMCQAInput = {
            "question": row["question"],
            "options": "\n".join(f"{k}. {v}" for k, v in zip(
                ["A", "B", "C", "D"],
                [row["opa"], row["opb"], row["opc"], row["opd"]]
            ))
        }
        label: MedMCQAOutput = ["A", "B", "C", "D"][row["cop"]] # wtf
        return input, label

    def evaluate_output(self, label: MedMCQAOutput, prediction: MedMCQAOutput | None) -> MedMCQAEvalResult:
        result = label == prediction
        return result

class SupportsMedMCQA(SupportsBenchmark[MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult]):
    ...