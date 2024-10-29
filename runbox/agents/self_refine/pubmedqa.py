from ._agent import SelfRefineAgent
from runbox.benchmarks.benchmarks.pubmedqa import PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult, SupportsPubMedQA


class PubMedQASelfRefineAgent(SelfRefineAgent[PubMedQAInput, PubMedQAOutput, PubMedQAEvalResult]):
    def parse(self, extracted_str: str) -> PubMedQAOutput | None:
        return extracted_str
