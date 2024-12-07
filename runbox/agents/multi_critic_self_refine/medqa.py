from ._agent import MultiCriticSelfRefineAgent
from runbox.benchmarks.benchmarks.medqa import MedQAInput, MedQAOutput, MedQAEvalResult


class MedQAMultiCriticSelfRefineAgent(MultiCriticSelfRefineAgent[MedQAInput, MedQAOutput, MedQAEvalResult]):
    def parse(self, extracted_str: str) -> MedQAOutput | None:
        return extracted_str

    @staticmethod
    def calc_full_score(full: list[list[MedQAEvalResult]]) -> list[float]:
        return [*map(lambda x: sum(x) / len(x), zip(*full))]