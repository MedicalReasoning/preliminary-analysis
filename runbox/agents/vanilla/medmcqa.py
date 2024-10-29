from ._agent import VanillaAgent
from runbox.benchmarks.benchmarks.medmcqa import MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult, SupportsMedMCQA


class MedMCQAVanillaAgent(VanillaAgent[MedMCQAInput, MedMCQAOutput, MedMCQAEvalResult]):
    def parse(self, extracted_str: str) -> MedMCQAOutput | None:
        return extracted_str