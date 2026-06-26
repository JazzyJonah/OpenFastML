import ROOT
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import random
from functools import partial
import time
# import memory_profiler

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from fastml.utils.image import augment_batch
from egammaCNN.model import build_qcnn
from tensorflow.keras import callbacks
from tensorflow.data import AUTOTUNE


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch_times.append(elapsed)
        print(f"Epoch {epoch + 1} took {elapsed:.6f} seconds")

def reformat_batch(x, y, w):
    shape = [3,3,2]

    leading_shape = tf.shape(x)[:-1]
    x = tf.reshape(x, tf.concat([leading_shape, shape], axis=0))
    y = tf.reshape(y, tf.concat([leading_shape, [1,1,1]], axis=0))
    return x, y, w

def reshape_dataset(ds, new_shape=[3,3,2]): # May be marginally faster--I'm not sure
    """
    Reshape the first element of each dataset item from (..., flat_dim)
    to (..., *new_shape), leaving the other elements unchanged.
    """

    # Get the original flattened dimension from the dataset spec
    flat_dim = ds.element_spec[0].shape[-1]

    if flat_dim is not None and np.prod(new_shape) != flat_dim:
        raise ValueError(
            f"Product of new_shape {new_shape} is {np.prod(new_shape)}, "
            f"but flattened dimension is {flat_dim}."
        )

    def _reshape(x, y, z):
        leading_shape = tf.shape(x)[:-1]
        x = tf.reshape(x, tf.concat([leading_shape, new_shape], axis=0))
        y = tf.reshape(y, tf.concat([leading_shape, [1,1,1]], axis=0))
        return x, y, z

    return ds.map(_reshape)

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

# @profile
def train_model(
        epochs=45, 
        max_trials=3, 
        project="root_qcnn", 
        batch_size=64,
        batches_in_memory=1000,
        drop_remainder=False,
        load_eager=False,
        shuffle=False,
        set_seed=42,
):
    print(f"This run is for project {project}", flush=True)

    df_train = ROOT.RDataFrame("tree", "train_data/train_data.root")
    df_val = ROOT.RDataFrame("tree", "train_data/val_data.root")
    print("Data loaded")

    train_columns = ["x_train", "y_train", "w_train"]
    val_columns = ["x_val", "y_val", "w_val"]
    train_max_vec_sizes = {"x_train": 3*3*2}
    val_max_vec_sizes = {"x_val": 3*3*2}


    train_dl = ROOT.Experimental.ML.RDataLoader(
        df_train,
        batch_size=batch_size,
        batches_in_memory=batches_in_memory,
        drop_remainder=drop_remainder,
        columns=train_columns,
        target="y_train",
        weights="w_train",
        max_vec_sizes=train_max_vec_sizes,
        load_eager=load_eager,
        set_seed=set_seed,
        shuffle=shuffle,
    )
    val_dl = ROOT.Experimental.ML.RDataLoader(
        df_val,
        batch_size=batch_size,
        batches_in_memory=batches_in_memory,
        drop_remainder=drop_remainder,
        columns=val_columns,
        target="y_val",
        weights="w_val",
        max_vec_sizes=val_max_vec_sizes,
        load_eager=load_eager,
        set_seed=set_seed,
        shuffle=shuffle,
    )

    train_ds = (
        train_dl.as_tensorflow()
        .map(reformat_batch, num_parallel_calls=AUTOTUNE)
        .map(augment_batch, num_parallel_calls=AUTOTUNE)
        .cache() # Only needs to run once, instead of every epoch.
        .prefetch(AUTOTUNE)
    )
    # train_ds = reshape_dataset(train_dl.as_tensorflow()).map(augment_batch, num_parallel_calls=AUTOTUNE)


    #### BATCH RETRIEVAL TESTING ####
    if True:
        def time_dataset(ds, n=1000, warmup=0):
            # Warmup: iterator/thread/file-cache startup, etc.
            for _ in ds.take(warmup):
                pass
            times = []
            start = time.perf_counter()
            
            for _ in ds.take(n):
                # print(i)
                # for j in i[0]:
                #     pass
                end = time.perf_counter()
                times.append(end-start)
                start = end

            return times

        times = time_dataset(train_ds)
        print(times, np.median(times))
        exit(0)

        x_batch, y_batch, w_batch = next(iter(raw_batched))

        cached_one_batch = (
            tf.data.Dataset
            .from_tensors((x_batch, y_batch, w_batch))
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

        baseline_hp = kt.HyperParameters()
        baseline_hp.Fixed("depth_mult", 4)
        baseline_hp.Fixed("conv_precision", 12)
        baseline_hp.Fixed("dense_precision", 12)
        baseline_hp.Fixed("learning_rate", 5e-4)
        build_qcnn_fixed = partial(build_qcnn, layers=2) # Not sure what the logic is doing, but the value is 2
        model = build_qcnn_fixed(baseline_hp)
        timer = EpochTimer()
        steps = 1000
        model.fit(cached_one_batch, steps_per_epoch=steps, epochs=3, callbacks=[timer], verbose=0)
        train_s_per_step = timer.epoch_times[-1] / steps
        print(f"training mostly only: {train_s_per_step * 1000:.3f} ms/step")
        model.fit(
            cached_one_batch,
            steps_per_epoch=steps,
            epochs=3,
            callbacks=[timer],
            verbose=0,
        )

        print("=====SUMMARY=====", flush=True)
        print(f"input only:   {input_s_per_batch * 1000:.3f} ms/batch")
        print(f"train only:   {train_s_per_step * 1000:.3f} ms/step")
        exit()
    #################################




    val_ds = (
        val_dl.as_tensorflow()
        .map(reformat_batch, num_parallel_calls=AUTOTUNE)
        # .cache() # Only needs to run once, instead of every epoch.
        .prefetch(AUTOTUNE)
    )
    # val_ds = reshape_dataset(val_dl.as_tensorflow())


    model_save_dir = os.path.join("models", project)
    os.makedirs(model_save_dir, exist_ok=True)

    # baseline
    baseline_hp = kt.HyperParameters()
    baseline_hp.Fixed("depth_mult", 4)
    baseline_hp.Fixed("conv_precision", 12)
    baseline_hp.Fixed("dense_precision", 12)
    baseline_hp.Fixed("learning_rate", 5e-4)

    build_qcnn_fixed = partial(build_qcnn, layers=2) # Not sure what the logic is doing, but the value is 2
    baseline_model = build_qcnn_fixed(baseline_hp)
    timer = EpochTimer()
    history = baseline_model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True), 
                        timer
                        ])
    print(f"Training losses:   {history.history['loss']}")
    print(f"Validation losses: {history.history['val_loss']}")
    baseline_model.save(os.path.join(model_save_dir, f"baseline_model_{batch_size}.keras"))

    # tuner
    build_qcnn_fixed = partial(build_qcnn, layers= 2)
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
    return timer.epoch_times
if __name__ == "__main__":
    x = time.time()
    train_model(batch_size=8)
    print(f"The time it took to run this program was {time.time()-x:.3f} seconds.")
