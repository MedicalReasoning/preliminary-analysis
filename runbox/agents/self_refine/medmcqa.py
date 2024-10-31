from ._agent import SelfRefineAgent
from runbox.benchmarks.benchmarks.medmcqa import MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult


class MedMCQASelfRefineAgent(SelfRefineAgent[MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult]):
    def parse(self, extracted_str: str) -> MedMCQAOutput | None:
        return extracted_str

    @staticmethod
    def calc_full_score(full: list[list[MedMCQAEvalResult]]) -> list[float]:
        return [*map(lambda x: sum(x) / len(x), zip(*full))]
