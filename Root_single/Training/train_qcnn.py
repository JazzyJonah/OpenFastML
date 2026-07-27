import ROOT
import os

from functools import partial
import time

import keras_tuner as kt
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

def reformat_batch(x, y, w):
    shape = [3,3,2]

    leading_shape = tf.shape(x)[:-1]
    x = tf.reshape(x, tf.concat([leading_shape, shape], axis=0))
    y = tf.reshape(y, tf.concat([leading_shape, [1,1,1]], axis=0))
    return x, y, w

def make_datasets(
        batch_size: int=512,
        batches_in_memory: int=1000,
        drop_remainder: bool=False,
        load_eager: bool=False,
        shuffle: bool=True,
        set_seed: int=1
) -> tuple[Dataset]:
    
    tf.keras.utils.set_random_seed(set_seed)

    columns = ["x", "y", "w"]
    max_vec_sizes = {"x": 3*3*2}
    df = ROOT.RDataFrame("tree", "Root_single/Data/processed.root")
    df_sig = df.Filter("y == 1")
    df_bkg = df.Filter("y == 0")
    if load_eager:
        dfs = df
    else:
        dfs = [df_sig, df_bkg]
    dl = ROOT.Experimental.ML.RDataLoader(
        dfs,
        batch_size=batch_size,
        batches_in_memory=batches_in_memory,
        drop_remainder=drop_remainder,
        columns=columns,
        target="y",
        weights="w",
        max_vec_sizes=max_vec_sizes,
        load_eager=load_eager,
        set_seed=set_seed,
        shuffle=shuffle
    )
    train_dl, val_dl = dl.train_test_split(test_size=0.2)

    train_ds = (
        train_dl.as_tensorflow()
        .map(reformat_batch, num_parallel_calls=AUTOTUNE)
        .map(augment_batch, num_parallel_calls=AUTOTUNE)
        .cache() # Only needs to run once, instead of every epoch.
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_dl.as_tensorflow()
        .map(reformat_batch, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds

    
def train_model(
        epochs: int=45, 
        max_trials: int=3, 
        batch_size: int=512,
        batches_in_memory: int=1000,
        drop_remainder: bool=False,
        load_eager: bool=False,
        shuffle: bool=True,
        set_seed: int=1,
        save: bool=True,
        save_path: str="",
        verbose: int=2
) -> tuple[tf.keras.callbacks.History, EpochTimer]:

    tf.keras.utils.set_random_seed(set_seed)
    train_ds, val_ds = make_datasets(
        batch_size,
        batches_in_memory,
        drop_remainder,
        load_eager,
        shuffle,
        set_seed
    )

    
    model_save_dir = "Root_single/Models"
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
        directory="Root_single/Models/",
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
