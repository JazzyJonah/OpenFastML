from time import time
import csv

def time_preprocessing(workflow):
    start = time()
    dataset = workflow()
    dataset.load()
    return time()-start

def write_preprocessing_time(
        directories: list[str],
        save_path: str="Benchmarks/Data/preprocessing_time.csv"
):
    times: dict[str, float] = {}
    for directory in directories:
        if directory == "Original":
            from Original.Preprocessing.trainingdataloader import OpenDataSet as original
            times[directory] = time_preprocessing(original)
        elif directory == "Hybrid":
            from Hybrid.Preprocessing.trainingdataloader import OpenDataSet as hybrid
            times[directory] = time_preprocessing(hybrid)
        elif directory == "ROOT":
            from Root.Preprocessing.trainingdataloader import RootDataSet as root
            times[directory] = time_preprocessing(root)
        elif directory == "ROOT_single":
            from Root_single.Preprocessing.trainingdataloader import RootDataSet as single
            times[directory] = time_preprocessing(single)

    try:
        with open(save_path, newline='') as f:
            oldTimes = next(csv.DictReader(f))
    except:
        oldTimes = {}
    times = oldTimes | times

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=times)
        writer.writeheader()
        writer.writerow(times)

if __name__ == "__main__":
    write_preprocessing_time(["Original", "Hybrid", "ROOT", "ROOT_single"])