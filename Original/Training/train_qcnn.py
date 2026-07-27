import os

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

def make_datasets(
        batch_size: int=512,
        set_seed: int=0
) -> tuple[Dataset]:
    
    tf.keras.utils.set_random_seed(set_seed)

    train = ak.from_parquet("Original/Data/train.parquet")
    val = ak.from_parquet("Original/Data/val.parquet")

    train_ds = (
        Dataset.from_tensor_slices((train.x_train, train.y_train, train.w_train))
        .batch(batch_size)
        .map(augment_batch, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        Dataset.from_tensor_slices((val.x_val, val.y_val, val.w_val))
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds

def train_model(
        epochs: int=45,
        max_trials: int=3,
        batch_size: int=512,
        set_seed: int=0,
        save: bool=True,
        save_path: str="",
        verbose: int=2
) -> tuple[tf.keras.callbacks.History, EpochTimer]:
    
    tf.keras.utils.set_random_seed(set_seed)
    train_ds, val_ds = make_datasets(
        batch_size, 
        set_seed
    )

    model_save_dir = "Original/Models"
    if not save_path:
        save_path = f"model_{batch_size}.keras"

    # baseline
    baseline_hp = kt.HyperParameters()
    baseline_hp.Fixed("depth_mult", 4)
    baseline_hp.Fixed("conv_precision", 12)
    baseline_hp.Fixed("dense_precision", 12)
    baseline_hp.Fixed("learning_rate", 5e-4)

    build_qcnn_fixed = partial(build_qcnn, layers=2)
    baseline_model = build_qcnn_fixed(baseline_hp)
    timer = EpochTimer()
    history = baseline_model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=[callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True), 
                        timer
                        ])
    if save:
        baseline_model.save(os.path.join(model_save_dir, save_path))

    # tuner
    build_qcnn_fixed = partial(build_qcnn, layers=2)
    tuner = kt.RandomSearch(
        build_qcnn_fixed,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=False,
        directory="Original/Models/",
        project_name="tuner_info",
        seed=set_seed,
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
        ],
    )

    if verbose > 0:
        tuner.results_summary()
    best_model = tuner.get_best_models(1)[0]
    if save:
        best_model.save(os.path.join(model_save_dir, "tuner_model.keras"))
    return history, timer
if __name__ == "__main__":
    x = time.time()
    train_model()
    print(f"The time it took to run this program was {time.time()-x:.3f} seconds.")
