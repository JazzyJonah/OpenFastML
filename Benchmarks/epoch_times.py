import csv

def time_epochs(train_func, n):
    times = []
    for _ in range(n):
        times.append(train_func(verbose=0, save=False)[1].epoch_times)
    return times

def write_epoch_times(
        directories: list[str],
        iterations: int=50,
        save_path: str="Benchmarks/Data/epoch_times.csv"
):
    times: dict[str, list[list[float]]] = {}
    for directory in directories:
        if directory == "Original":
            from Original.Training.train_qcnn import train_model as original
            times[directory] = time_epochs(original, iterations)
        elif directory == "Hybrid":
            from Hybrid.Training.train_qcnn import train_model as hybrid
            times[directory] = time_epochs(hybrid, iterations)
        elif directory == "ROOT":
            from Root.Training.train_qcnn import train_model as root
            times[directory] = time_epochs(root, iterations)
        elif directory == "ROOT_single":
            from Root_single.Training.train_qcnn import train_model as single
            times[directory] = time_epochs(single, iterations)

    try:
        with open(save_path, newline="") as f:
            reader = csv.DictReader(f)
            old_headers = reader.fieldnames or []
            old = {(workflow["run"], workflow["epoch"]): workflow for workflow in reader}
    except FileNotFoundError:
        old_headers, old = [], {}

    rows = [
        old.get((str(run), str(epoch)), {}) | {
            "run": run,
            "epoch": epoch,
            **{header: values[run][epoch] for header, values in times.items()},
        }
        for run in range(iterations)
        for epoch in range(45)
    ]

    headers = old_headers + [
        header for header in ["run", "epoch", *times]
        if header not in old_headers
    ]

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    write_epoch_times(["Original", "Hybrid", "ROOT", "ROOT_single"])