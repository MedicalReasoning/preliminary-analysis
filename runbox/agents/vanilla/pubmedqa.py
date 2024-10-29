from ._agent import VanillaAgent
from runbox.benchmarks.benchmarks.pubmedqa import PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult, SupportsPubMedQA


class PubMedQAVanillaAgent(VanillaAgent[PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult]):
    def parse(self, extracted_str: str) -> PubMedQAOutput | None:
        return extracted_str