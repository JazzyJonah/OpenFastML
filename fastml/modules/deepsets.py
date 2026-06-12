from tensorflow.keras import Input, layers, Sequential
from qkeras import quantized_bits, QActivation, QDense
from qkeras.utils import load_qmodel

from fastml.modules.layers import WeightedSum


def QuantisedDeepSets(
    input_shape,
    embed_dim,
    target_dim,
    quantisation,
    output_activation=layers.Activation("hard_sigmoid"),
    **kwargs,
):
    bits, integer = quantisation

    quantiser = quantized_bits(bits=bits, integer=integer)

    model = Sequential(**kwargs)
    model.add(Input(shape=input_shape))

    for d in embed_dim:
        model.add(QDense(d, kernel_quantizer=quantiser, bias_quantizer=quantiser))
        model.add(QActivation(f"quantized_relu({bits},{integer})"))

    model.add(layers.GlobalAveragePooling1D())

    for i, d in enumerate(target_dim):
        model.add(QDense(d, kernel_quantizer=quantiser, bias_quantizer=quantiser))

        if i == len(target_dim) - 1:
            model.add(output_activation)

        else:
            model.add(QActivation(f"quantized_relu({bits},{integer})"))

    model.summary()
    return model


def load_qDeepSets(path):
    return load_qmodel(
        path,
        custom_objects={
            "WeightedSum": WeightedSum,
            "QuantisedDeepSets": QuantisedDeepSets,
        },
    )
