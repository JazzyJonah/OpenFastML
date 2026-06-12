from tensorflow.keras import layers, Input, backend, Model


def MLPCalibrator(
    hidden_layer_sizes: list[int] = [32, 32, 32],
    hidden_activation: str = "relu",
    max_jets: int = 20,
    eps: float = 1e-3,
    scale_factor_activation="exponential",
    initialise_as_identity=True,
):
    pt = Input(shape=(max_jets, 1), name="pt")
    abseta = Input(shape=(max_jets, 1), name="abseta")

    # precompute log
    log_pt = backend.log(pt + eps)
    log_abseta = backend.log(abseta + eps)

    x = layers.Concatenate()([log_pt, log_abseta])

    for hls in hidden_layer_sizes:
        x = layers.Dense(hls, activation=hidden_activation)(x)

    scale_factor = layers.Dense(
        1,
        activation=scale_factor_activation,
        bias_initializer="zeros",
        kernel_initializer="zeros" if initialise_as_identity else "glorot_uniform",
    )(x)

    calib_pt = pt * scale_factor

    return Model(inputs=[pt, abseta], outputs=[calib_pt])
