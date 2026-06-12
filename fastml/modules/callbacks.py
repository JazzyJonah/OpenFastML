from tensorflow.keras import callbacks
import numpy as np


def is_symmetric(model, axes=(1, 2)):
    input_shape = [100, 3, 3, model.input.shape[-1]]
    inputs = np.random.randn(*input_shape)
    test_vectors = [inputs] + [np.flip(inputs, axis=axis) for axis in axes]
    outputs = [model.predict(x, verbose=0) for x in test_vectors]
    return np.allclose(outputs, outputs[0])


class ConditionalCheckpoint(callbacks.Callback):
    def __init__(self, condition, filepath=None, monitor="val_loss"):
        super().__init__()

        self.condition = condition
        self.filepath = filepath
        self.monitor = monitor

    def on_train_begin(self, logs=None):
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best) and self.condition(self.model):
            self.best = current

            if self.filepath is not None:
                self.model.save(self.filepath, overwrite=True)
                print(f"Saving model to {self.filepath}")

        logs[f"min_{self.monitor}"] = self.best
