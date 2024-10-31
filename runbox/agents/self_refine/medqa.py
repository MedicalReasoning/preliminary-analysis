from ._agent import SelfRefineAgent
from runbox.benchmarks.benchmarks.medqa import MedQAInput, MedQAOutput, MedQAEvalResult


class MedQASelfRefineAgent(SelfRefineAgent[MedQAInput, MedQAOutput, MedQAEvalResult]):
    def parse(self, extracted_str: str) -> MedQAOutput | None:
        return extracted_str

    @staticmethod
    def calc_full_score(full: list[list[MedQAEvalResult]]) -> list[float]:
        return [*map(lambda x: sum(x) / len(x), zip(*full))]