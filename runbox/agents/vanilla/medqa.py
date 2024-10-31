from ._agent import VanillaAgent
from runbox.benchmarks.benchmarks.medqa import MedQAInput, MedQAOutput, MedQAEvalResult


class MedQAVanillaAgent(VanillaAgent[MedQAInput, MedQAOutput, MedQAEvalResult]):
    def parse(self, extracted_str: str) -> MedQAOutput | None:
        return extracted_str