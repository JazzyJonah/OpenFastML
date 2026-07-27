from fastml.modules.cnn import QuantisedClasCNN
from tensorflow.keras import losses, Input, Model, optimizers

def build_qcnn(hp, layers = 6):
    input = Input(shape=(None, None, layers))
    x = input
    x = QuantisedClasCNN(
        depth_multiplier=hp.Choice("depth_mult", [2, 3, 4]),
        conv_precision=hp.Int("conv_precision", min_value=6, max_value=12, step=2),
        dense_precision=hp.Int("dense_precision", min_value=6, max_value=12, step=2),
        input_shape = (None, None, layers),
        name="qdcnn",
    )(x)

    model = Model(input, x)

    lr = hp.Choice("learning_rate", [1e-2, 1e-3])

    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(
            learning_rate= lr
        ),
        weighted_metrics=[],
    )

    model.summary()

    return model
