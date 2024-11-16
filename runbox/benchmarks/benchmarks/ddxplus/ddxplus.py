from typing import TypedDict, TypeVar
import json

from ...benchmark import Benchmark


class DDXPlusInput(TypedDict):
    evidences: str
    options: str

type DDXPlusOutput = str

type DDXPlusEvalResult = bool


_release_evidences = json.load(open("runbox/benchmarks/benchmarks/ddxplus/release_evidences.json"))


def _load_evidences(evidences: str) -> list[str]:
    return json.loads(evidences.replace("'", '"'))

def _convert_answer(row: dict, value: int | None) -> str:
    if row["data_type"] == "B":
        return "N" if value == 0 else "Y"
    elif row["data_type"] == "C":
        return f"{value} (possible values: {row['possible-values']})"
    else:
        raise Exception()

def _generate_str(row: dict) -> str:
    evidences = _load_evidences(row["EVIDENCES"])
    ev = ""
    buf = [
        f"Age: {row['AGE']}",
        f"Sex: {row['SEX']}"
    ]
    for code in evidences:
        code_split = code.split("_@_")
        if code_split[0] != ev:
            ev = code_split[0]
            buf.append(_release_evidences[code_split[0]]["question_en"])
        if len(code_split) > 1:
            if "V" not in code_split[1]:
                ans = _convert_answer(
                    _release_evidences[code_split[0]],
                    int(code_split[1])
                )
                buf.append(f'    - {ans}')
            else:
                buf.append(f'    - {_release_evidences[code_split[0]]["value_meaning"][code_split[1]]["en"]}')
        else:
            ans = _convert_answer(
                _release_evidences[code_split[0]],
                _release_evidences[code_split[0]]["default_value"]
            )
            buf.append(f'    - {ans}')
    return "\n".join(buf)


class DDXPlus(Benchmark[DDXPlusInput, DDXPlusOutput, DDXPlusEvalResult]):
    def __init__(
        self,
        split: str,
        slice: tuple[int, int] | None = None
    ) -> None:
        assert split in ["train", "validate", "test"], "invalid `split`."
        super().__init__(
            "aai530-group6/ddxplus",
            split=split,
            slice=slice
        )

    def preprocess_row(self, row: dict) -> tuple[DDXPlusInput, DDXPlusOutput]:
        input: DDXPlusInput = {
            "evidences": _generate_str(row),
            "options": "\n".join(f"- {o[0]}" for o in json.loads(row["DIFFERENTIAL_DIAGNOSIS"].replace("'", '"')))
        }
        label: DDXPlusOutput = row["PATHOLOGY"]
        return input, label

    def evaluate_output(self, label: DDXPlusOutput, prediction: DDXPlusOutput | None) -> DDXPlusEvalResult:
        if label is not None:
            label = label.strip().lower()
        if prediction is not None:
            prediction = prediction.strip().lower()
        result = label == prediction
        return result