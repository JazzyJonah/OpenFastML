from time import time
import csv

def time_batching(dataset_func, n, warmup=2):
    ds = dataset_func()[0].repeat()
    times = []
    start = time()
    for _ in ds.take(n):
        end = time()
        times.append(end-start)
        start = end
    return times[warmup:]

def write_batching_times(
        directories: list[str],
        iterations: int=1000,
        save_path: str="Benchmarks/Data/batch_fetch_times.csv"
):
    times: dict[str, list[float]] = {}
    for directory in directories:
        if directory == "Original":
            from Original.Training.train_qcnn import make_datasets as original
            times[directory] = time_batching(original, iterations)
        elif directory == "Hybrid":
            from Hybrid.Training.train_qcnn import make_datasets as hybrid
            times[directory] = time_batching(hybrid, iterations)
        elif directory == "ROOT":
            from Root.Training.train_qcnn import make_datasets as root
            times[directory] = time_batching(root, iterations)
        elif directory == "ROOT_single":
            from Root_single.Training.train_qcnn import make_datasets as single
            times[directory] = time_batching(single, iterations)

    try:
        with open(save_path, newline="") as f:
            old = list(csv.DictReader(f))
    except FileNotFoundError:
        old = [{}] * iterations

    rows = [
        old_row | dict(zip(times, values))
        for old_row, values in zip(old, zip(*times.values()))
    ]

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    write_batching_times(["Original", "Hybrid", "ROOT", "ROOT_single"])