import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import random
from functools import partial
import time

import awkward as ak
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from fastml.utils.image import augment_batch
from egammaCNN.model import build_qcnn
from tensorflow.keras import callbacks
from tensorflow.data import Dataset, AUTOTUNE


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch_times.append(elapsed)
        print(f"Epoch {epoch + 1} took {elapsed:.6f} seconds")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except AttributeError:
    pass
try:
    tf.keras.utils.set_random_seed(SEED)
except AttributeError:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options({"deterministic": True})
except Exception:
    pass

x = time.time()

epochs = 45
max_trials = 3
project = f"cls_qcnn"

print(f"This run is for project {project}", flush=True)

train = ak.from_parquet("train_data/train_data.parquet")
val = ak.from_parquet("train_data/val_data.parquet")

print("Data loaded")

train_ds = (
    Dataset.from_tensor_slices((train.x_train, train.y_train, train.w_train))
    # .batch(len(train.y_train) // 512)
    .batch(64)
    # .map(lambda x, y, w: augment_batch(x, y, w), num_parallel_calls=AUTOTUNE)
    .map(augment_batch, num_parallel_calls=AUTOTUNE) # lambda expression not necessary
    .prefetch(AUTOTUNE)
)

# # BATCH RETRIEVAL TESTING
# for i in train_ds:
#     break
# start = time.perf_counter()
# for batch in train_ds.take(1000):
#     time.sleep(0.0005) # 500 microseconds; the lower bound for how long a step will take
# elapsed = time.perf_counter() - start - 0.0005 * 1000
# print(f"It took an avergae of {elapsed/100} seconds to retrieve a batch.")
# exit()

val_ds = (
    Dataset.from_tensor_slices((val.x_val, val.y_val, val.w_val))
    # .batch(len(val.y_val) // 512)
    .batch(64)
    .prefetch(AUTOTUNE)
)

model_save_dir = os.path.join("models", project)
os.makedirs(model_save_dir, exist_ok=True)

# baseline
baseline_hp = kt.HyperParameters()
baseline_hp.Fixed("depth_mult", 4)
baseline_hp.Fixed("conv_precision", 12)
baseline_hp.Fixed("dense_precision", 12)
baseline_hp.Fixed("learning_rate", 5e-4)

build_qcnn_fixed = partial(build_qcnn, layers= int(ak.to_numpy(train.x_train).shape[-1]))
baseline_model = build_qcnn_fixed(baseline_hp)
timer = EpochTimer()
baseline_model.fit(train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=2,
                callbacks=[callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True), 
                    timer
                    ])
print(timer.epoch_times)

baseline_model.save(os.path.join(model_save_dir, "baseline_model.keras"))

# tuner
build_qcnn_fixed = partial(build_qcnn, layers= int(ak.to_numpy(train.x_train).shape[-1]))
tuner = kt.RandomSearch(
    build_qcnn_fixed,
    objective="val_loss",
    max_trials=max_trials,
    overwrite=False,
    directory="models/",
    project_name=project,
    seed=0,
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ],
)

tuner.results_summary()
best_model = tuner.get_best_models(1)[0]
best_model.save(os.path.join(model_save_dir, "tuner_model.keras"))

print(f"The time it took to run this program was {time.time()-x:.3f} seconds.")