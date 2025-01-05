from ._agent import GenRMAgent
from runbox.benchmarks.benchmarks.ddxplus import DDXPlusInput, DDXPlusOutput, DDXPlusEvalResult


class DDXPlusGenRMAgent(GenRMAgent[DDXPlusInput, DDXPlusOutput, DDXPlusEvalResult]):
    def parse(self, extracted_str: str) -> DDXPlusOutput | None:
        return extracted_str

    @staticmethod
    def calc_full_score(full: list[list[DDXPlusEvalResult]]) -> list[float]:
        return [*map(lambda x: sum(x) / len(x), zip(*full))]
