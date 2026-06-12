from tensorflow.keras import layers, Sequential, Input, backend, Model
import tensorflow as tf
from qkeras import quantized_bits, QActivation, QDense
from qkeras.utils import load_qmodel

from fastml.modules.layers import (
    WeightedSum,
    SymmetricPooling,
    QSymmetricDepthwiseConv2D,
    SymmetricDepthwiseConv2D,
)


backend.set_image_data_format("channels_last")


def QuantisedCNN(
    depth_multiplier,
    conv_precision,
    dense_precision,
    input_shape=(None, None, 6),
    output_activation=layers.Activation("hard_sigmoid"),
    name="qcnn",
):
    conv_quantiser = quantized_bits(
        bits=conv_precision, integer=conv_precision // 2, alpha=1
    )
    dense_quantiser = quantized_bits(
        bits=dense_precision, integer=dense_precision // 2, alpha=1
    )

    cnn = Sequential(name=name)
    cnn.add(Input(shape=input_shape))
    cnn.add(
        QSymmetricDepthwiseConv2D(
            input_channels=input_shape[-1],
            depth_multiplier=depth_multiplier,
            kernel_quantizer=conv_quantiser,
            bias_quantizer=conv_quantiser,
        )
    )
    cnn.add(QActivation(f"quantized_relu({conv_precision},{conv_precision//2})"))
    cnn.add(
        QDense(
            6,
            kernel_quantizer=dense_quantiser,
            bias_quantizer=dense_quantiser,
        )
    )
    cnn.add(output_activation)

    return cnn

class ScalarGate(layers.Layer):
    def __init__(self, init=0.1, **kw):
        super().__init__(**kw)
        self.init = init
    def build(self, _):
        self.alpha = self.add_weight(
            "alpha", shape=(), initializer=tf.keras.initializers.Constant(self.init)
        )
    def call(self, x):
        return x * tf.nn.softplus(self.alpha)

def QuantisedTwoBranchCNN(
    depth_multiplier,
    conv_precision,
    dense_precision,
    cnn_in=(3, 3, 6),
    kin_in=7,
    name="egamma_qcnn_two_branch",
):
    conv_q  = quantized_bits(bits=conv_precision,  integer=conv_precision // 2,  alpha=1)
    dense_q = quantized_bits(bits=dense_precision, integer=dense_precision // 2, alpha=1)

    x_in = Input(shape=cnn_in, name="x_in")
    x = QSymmetricDepthwiseConv2D(
            input_channels=cnn_in[-1], depth_multiplier=depth_multiplier,
            kernel_quantizer=conv_q, bias_quantizer=conv_q, name="dwconv")(x_in)
    x = QActivation(f"quantized_relu({conv_precision},{conv_precision//2})")(x)
    x = layers.Flatten(name="flat")(x)    
    x = QDense(16, kernel_quantizer=dense_q, bias_quantizer=dense_q, name="shape_emb")(x)
    x = QActivation(f"quantized_relu({conv_precision},{conv_precision//2})")(x)

    k_in = Input(shape=(kin_in,), name="k_in")                           # [pT, 6 fractions]
    k = QDense(12, kernel_quantizer=dense_q, bias_quantizer=dense_q,
               kernel_regularizer=tf.keras.regularizers.l2(1e-2), name="kin_fc1")(k_in)
    k = QActivation(f"quantized_relu({dense_precision},{dense_precision//2})")(k)
    k = QDense(4, kernel_quantizer=dense_q, bias_quantizer=dense_q,
               kernel_regularizer=tf.keras.regularizers.l2(1e-2), name="kin_fc2")(k)
    k = QActivation(f"quantized_relu({dense_precision},{dense_precision//2})")(k)
    k = ScalarGate(init=0.1, name="kin_gate")(k)    

    h = layers.Concatenate(name="branch_concat")([x, k])
    logit = QDense(1, kernel_quantizer=dense_q, bias_quantizer=dense_q, name="logit")(h)
    logit4d = tf.keras.layers.Reshape((1, 1, 1), name="logit4d")(logit)
    out = tf.keras.layers.Activation("sigmoid", name="prob")(logit4d)

    model = Model(inputs=[x_in, k_in], outputs=out, name=name)
    return model


def QuantisedClasCNN(
    depth_multiplier,
    conv_precision,
    dense_precision,
    input_shape=6,
    output_activation=layers.Activation("sigmoid"),
    name="clas_qcnn",
):
    conv_quantiser = quantized_bits(
        bits=conv_precision, integer=conv_precision // 2, alpha=1
    )
    dense_quantiser = quantized_bits(
        bits=dense_precision, integer=dense_precision // 2, alpha=1
    )

    cnn = Sequential(name=name)
    cnn.add(Input(shape=input_shape))
    cnn.add(
        QSymmetricDepthwiseConv2D(
            input_channels=input_shape[-1],
            depth_multiplier=depth_multiplier,
            kernel_quantizer=conv_quantiser,
            bias_quantizer=conv_quantiser,
        )
    )
    cnn.add(QActivation(f"quantized_relu({conv_precision},{conv_precision//2})"))
    cnn.add(
        QDense(
            1,
            kernel_quantizer=dense_quantiser,
            bias_quantizer=dense_quantiser,
        )
    )
    cnn.add(output_activation)

    return cnn

def QMVarClasCNN_extraDense(
    depth_multiplier,
    conv_precision,
    dense_precision,
    cnn_shape=(None, None, 6),
    var_dim=2,
    name="clas_qcnn",
):
    conv_quantiser = quantized_bits(
        bits=conv_precision, integer=conv_precision // 2, alpha=1
    )
    dense_quantiser = quantized_bits(
        bits=dense_precision, integer=dense_precision // 2, alpha=1
    )

    cnn_input = Input(shape=cnn_shape, name="cnn_in")

    cnn = QSymmetricDepthwiseConv2D(
        input_channels=cnn_shape[-1],
        depth_multiplier=depth_multiplier,
        kernel_quantizer=conv_quantiser,
        bias_quantizer=conv_quantiser,
    )(cnn_input)

    cnn = QActivation(f"quantized_relu({conv_precision},{conv_precision//2})")(cnn)

    var_input = Input(shape=(None, None, var_dim), name="var_in")

    conc = layers.Concatenate(name="branch_concat")([cnn, var_input])

    conc = QDense(
        13,
        kernel_quantizer=dense_quantiser,
        bias_quantizer=dense_quantiser,
    )(conc)

    conc = QActivation(f"quantized_relu({dense_precision},{dense_precision//2})")(conc)

    conc = QDense(
        1,
        kernel_quantizer=dense_quantiser,
        bias_quantizer=dense_quantiser,
    )(conc)

    out = layers.Activation("sigmoid")(conc)

    model = Model(inputs=[cnn_input, var_input], outputs=out, name=name)
    return model

def DualQuantisedCNN(
    num_classes,
    depth_multiplier,
    conv_precision,
    dense_precision,
    input_shape=(None, None, 6),
    output_activation1=layers.Activation("hard_sigmoid"),
    output_activation2=layers.Activation("softmax"),
    name="dual_qcnn",
):
    conv_quantiser = quantized_bits(
        bits=conv_precision, integer=conv_precision // 2, alpha=1
    )
    dense_quantiser = quantized_bits(
        bits=dense_precision, integer=dense_precision // 2, alpha=1
    )

    inputs = Input(shape=input_shape)
    x = inputs
    x = QSymmetricDepthwiseConv2D(
        input_channels=input_shape[-1],
        depth_multiplier=depth_multiplier,
        kernel_quantizer=conv_quantiser,
        bias_quantizer=conv_quantiser,
    )(x)
    x = QActivation(f"quantized_relu({conv_precision},{conv_precision//2})")(x)

    x = QDense(
        32,
        kernel_quantizer=dense_quantiser,
        bias_quantizer=dense_quantiser,
    )(x)
    x = QActivation(f"quantized_relu({dense_precision},{dense_precision//2})")(x)

    w = QDense(
        6,
        kernel_quantizer=dense_quantiser,
        bias_quantizer=dense_quantiser,
    )(x)

    w = output_activation1(w)

    y = QDense(
        num_classes,
        kernel_quantizer=dense_quantiser,
        bias_quantizer=dense_quantiser,
    )(x)

    y = output_activation2(y)

    model = Model(inputs=inputs, outputs=[w, y], name=name)

    return model


def load_qCNN(path):
    return load_qmodel(
        path,
        custom_objects={
            "WeightedSum": WeightedSum,
            "QuantisedClasCNN": QuantisedClasCNN,
            "QMVarClasCNN_extraDense": QMVarClasCNN_extraDense,
            "SymmetricPooling": SymmetricPooling,
            "QSymmetricDepthwiseConv2D": QSymmetricDepthwiseConv2D,
            "SymmetricDepthwiseConv2D": SymmetricDepthwiseConv2D,
        },
    )