from fastml.utils.image import augment_batch
from egammaCNN.model import build_qcnn
from tensorflow.keras import callbacks
from tensorflow.data import Dataset, AUTOTUNE
from functools import partial
import keras_tuner as kt
import awkward as ak
import os

epochs = 45
max_trials = 3
project = f"cls_qcnn"

print(f"This run is for project {project}", flush=True)

train = ak.from_parquet("train_data.parquet")
val = ak.from_parquet("val_data.parquet")

print("Data loaded")

train_ds = (
    Dataset.from_tensor_slices((train.x_train, train.y_train, train.w_train))
    .batch(len(train.y_train) // 512)
    .map(lambda x, y, w: augment_batch(x, y, w), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    Dataset.from_tensor_slices((val.x_val, val.y_val, val.w_val))
    .batch(len(val.y_val) // 512)
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
baseline_model.fit(train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=2,
                callbacks=[callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True)])

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