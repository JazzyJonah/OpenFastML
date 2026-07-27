from time import time
import csv

def time_training(train_func, n):
    times = []
    for _ in range(n):
        start = time()
        train_func(verbose=0, save=False)
        times.append(time()-start)
    return times

def write_training_times(
        directories: list[str],
        iterations: int=50,
        save_path: str="Benchmarks/Data/training_times.csv"
):
    times: dict[str, list[float]] = {}
    for directory in directories:
        if directory == "Original":
            from Original.Training.train_qcnn import train_model as original
            times[directory] = time_training(original, iterations)
        elif directory == "Hybrid":
            from Hybrid.Training.train_qcnn import train_model as hybrid
            times[directory] = time_training(hybrid, iterations)
        elif directory == "ROOT":
            from Root.Training.train_qcnn import train_model as root
            times[directory] = time_training(root, iterations)
        elif directory == "ROOT_single":
            from Root_single.Training.train_qcnn import train_model as single
            times[directory] = time_training(single, iterations)

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
    write_training_times(["Original", "Hybrid", "ROOT", "ROOT_single"])