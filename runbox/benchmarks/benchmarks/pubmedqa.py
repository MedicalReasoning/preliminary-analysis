from typing import TypedDict

from ..benchmark import Benchmark
from ..supports_benchmark import SupportsBenchmark


class PubMedQAInput(TypedDict):
    question: str
    context: str

type PubMedQAOutput = str

type PubMedQAEvalResult = bool

class PubMedQA(Benchmark[PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult]):
    def __init__(
        self,
        split: str,
        slice: tuple[int, int] | None = None
    ) -> None:
        assert split in ["train"], "invalid `split`."
        super().__init__(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split=split,
            slice=slice
        )

    def preprocess_row(self, row: dict) -> tuple[PubMedQAInput, PubMedQAOutput]:
        context = row["context"]

        input: PubMedQAInput = {
            "question": row["question"],
            "context": "\n".join(f"{k}: {v}" for k, v in zip(context["labels"], context["contexts"]))
        }
        label: PubMedQAOutput = row["final_decision"]
        return input, label

    def evaluate_output(self, label: PubMedQAOutput, prediction: PubMedQAOutput | None) -> PubMedQAEvalResult:
        result = label.lower() == (prediction if prediction is not None else "").lower()
        return result