from ._agent import SelfRefineAgent
from runbox.benchmarks.benchmarks.medmcqa import MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult, SupportsMedMCQA


class MedMCQASelfRefineAgent(SelfRefineAgent[MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult]):
    def parse(self, extracted_str: str) -> MedMCQAOutput | None:
        return extracted_str
