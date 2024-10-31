from ._agent import SelfRefineAgent
from runbox.benchmarks.benchmarks.pubmedqa import PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult


class PubMedQASelfRefineAgent(SelfRefineAgent[PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult]):
    def parse(self, extracted_str: str) -> PubMedQAOutput | None:
        return extracted_str

    @staticmethod
    def calc_full_score(full: list[list[PubMedQAEvalResult]]) -> list[float]:
        return [*map(lambda x: sum(x) / len(x), zip(*full))]
