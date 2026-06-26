import time
import subprocess
import sys
import statistics
from pathlib import Path
import numpy as np
import ROOT
# from memory_profiler import profile

from root_train_qcnn import train_model as train_root_importless
from train_qcnn import train_model as train_og_importless

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EGAMMA_DIR = PROJECT_ROOT / "egammaCNN"



og_code = f"""
import sys
sys.path.insert(0, {str(EGAMMA_DIR)!r})

from train_qcnn import train_model
train_model(max_trials=1)
"""
root_code = f"""
import sys
sys.path.insert(0, {str(EGAMMA_DIR)!r})

from root_train_qcnn import train_model
train_model(max_trials=1)
"""

# @profile
def train_root_imports():
    subprocess.run(
        [sys.executable, "-c", root_code],
         cwd=PROJECT_ROOT,
         check=True
    )

# @profile
def train_og_imports():
    subprocess.run(
        [sys.executable, "-c", og_code],
         cwd=PROJECT_ROOT,
         check=True
    )

# times = []
# for i in range(20):
#     print(f"Trial {i}")
#     start = time.perf_counter()
#     train_root_importless(max_trials=1, set_seed=i+1, batch_size=512)
#     times.append(time.perf_counter()-start)

# print(f"Mean running time:  {statistics.mean(times)}")
# print(f"Median:             {statistics.median(times)}")
# print(f"Standard Deviation: {statistics.stdev(times)}")
# print(f"Raw times:          {times}")

# medians = []
# batch_sizes = np.round(np.geomspace(8, 512, 20))
# for batch_size in batch_sizes:
#     batch_size = int(batch_size)
#     median = statistics.median(train_og_importless(batch_size=batch_size))
#     medians.append(median)
#     print(f"Current medians: {medians}")

df = ROOT.RDataFrame(5).Define("col", "rdfentry_").Define("othercol", "rdfentry_")
df = df.Filter("col > 2")
df.Display().Print()