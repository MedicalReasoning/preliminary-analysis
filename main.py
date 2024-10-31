from typing import TypedDict, Any
import json
import argparse
from multiprocessing import Queue, Process
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv # type: ignore
# from langchain_community.callbacks import get_openai_callback

from config import prepare


class BenchConfig(TypedDict):
    split: str
    slice: tuple[int, int]

class RunConfig(TypedDict):
    benchmark: str
    bench_config: BenchConfig
    models: tuple[str, str, str]

def generate_chunks(slice: tuple[int, int], n_process: int) -> list[tuple[int, int]]:
    r = (slice[1] - slice[0]) // n_process

    chunks = [
        (
            slice[0] + i*r,
            slice[0] + (i+1)*r
        )
        for i in range(n_process - 1)
    ]
    chunks.append((slice[0] + (n_process-1) * r, slice[1]))

    return chunks

def run_single_chunk(
    queue: Queue,
    config: RunConfig,
    chunk: tuple[int, int]
) -> None:
    _Benchmark, agent = prepare(config["benchmark"], *config["models"])
    dataset = _Benchmark(split=config["bench_config"]["split"], slice=chunk) # type: ignore

    results = []

    for input, label in dataset:
        try:
            output = agent.run(input)
            result = agent.evaluate(dataset.evaluate_output, label, output)
        except Exception as e:
            output = {}
            result = [False] * 4

        results.append({
            "input": input,
            "label": label,
            "output": output,
            "result": result
        })

    queue.put((chunk, results))

def calc_full_score(
    config: RunConfig,
    full: list[dict]
) -> Any:
    _, agent = prepare(config["benchmark"], *config["models"])
    return agent.calc_full_score(full)

def save_results(
    config: RunConfig,
    result_dir_path: str,
    result: dict
) -> None:
    file_name = f"{config['benchmark']}-{'-'.join(config['models'])}.json"
    file_path = Path(result_dir_path) / Path(file_name)
    with open(file_path, "w") as f:
        json.dump(result, f)

def run_single_config(
    config: RunConfig,
    n_process: int,
    result_dir_path: str
) -> None:
    chunks = generate_chunks(config["bench_config"]["slice"], n_process)

    processes: list[Process] = []
    queue = Queue()

    for chunk in chunks:
        p = Process(target=run_single_chunk, args=(queue, config, chunk))
        processes.append(p)
        p.start()

    results: list[tuple[tuple[int, int], list[dict]]] = []
    result_indexes: list[tuple[int, int, int]] = []
    for i, _ in enumerate(processes):
        result = queue.get()
        results.append(result)
        result_indexes.append((*result[0], i))
    sorted_indexes = sorted(result_indexes)
    results_ = [*map(lambda x: results[x[2]], sorted_indexes)]

    for p in processes:
        p.join()

    full: list[dict] = sum([*map(lambda x: x[1], results_)], [])
    full_score = calc_full_score(config, [*map(lambda x: x["result"], full)])

    save_results(
        config,
        result_dir_path,
        {
            "final_score": full_score,
            "generations": full
        }
    )

def load_queue(path: str) -> list[RunConfig]:
    queue: list[RunConfig] = json.load(open(path, "r"))
    return queue

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--queue_path", type=str)
    parser.add_argument("-n", "--n_process", type=int)
    parser.add_argument("-r", "--result_dir_path", type=str)

    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_args()
    queue_path: str = args.queue_path
    n_process: int = args.n_process
    result_dir_path: str = args.result_dir_path

    queue = load_queue(queue_path)

    for config in tqdm(queue, desc="configs"):
        run_single_config(config, n_process, result_dir_path)

if __name__ == "__main__":
    load_dotenv()
    main()