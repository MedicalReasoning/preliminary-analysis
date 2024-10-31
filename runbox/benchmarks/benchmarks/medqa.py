from typing import TypedDict

from ..benchmark import Benchmark
from ..supports_benchmark import SupportsBenchmark


class MedQAInput(TypedDict):
    question: str
    options: str

type MedQAOutput = str

type MedQAEvalResult = bool

class MedQA(Benchmark[MedQAInput, MedQAOutput, MedQAEvalResult]):
    def __init__(
        self,
        split: str,
        slice: tuple[int, int] | None = None
    ) -> None:
        assert split in ["train", "test"], "invalid `split`."
        super().__init__(
            "bigbio/med_qa",
            split=split,
            slice=slice,
            trust_remote_code=True
        )

    def preprocess_row(self, row: dict) -> tuple[MedQAInput, MedQAOutput]:
        input: MedQAInput = {
            "question": row["question"],
            "options": "\n".join(f"{option['key']}. {option['value']}" for option in row["options"])
        }
        label: MedQAOutput = row["answer_idx"]
        return input, label

    def evaluate_output(self, label: MedQAOutput, prediction: MedQAOutput | None) -> MedQAEvalResult:
        result = label == prediction
        return result