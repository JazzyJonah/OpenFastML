from tensorflow.keras import layers, backend, initializers
import keras
import tensorflow as tf
import numpy as np
from qkeras import QDense

from fastml.utils.cells import quantise, fixed_encoding, fixed_decoding


class FixedEncodingLayer(layers.Layer):
    def __init__(self):
        super(layers.Layer, self).__init__()

    def call(self, inputs):
        return fixed_encoding(inputs)


class FixedDecodingLayer(layers.Layer):
    def __init__(self):
        super(layers.Layer, self).__init__()

    def call(self, inputs):
        return fixed_decoding(inputs)


class QuantisationLayer(layers.Layer):
    def __init__(self):
        super(layers.Layer, self).__init__()

    def call(self, inputs):
        return quantise(inputs)


class WeightedSum(layers.Layer):
    def call(self, x, weights):
        return backend.sum(x * weights, axis=-1, keepdims=True)


class RandomSymmetricKernel(initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        k = initializers.GlorotUniform()(shape, dtype=dtype)
        aug_k = tf.math.reduce_mean(
            [
                k,
                tf.reverse(k, axis=[0]),
                tf.reverse(k, axis=[1]),
                tf.reverse(k, axis=[0, 1]),
            ],
            axis=0,
        )
        return aug_k


class SymmetricPooling(layers.Layer):
    def __init__(self, input_channels):
        super().__init__()

        k = np.zeros((3, 3, 1, 3), dtype=np.float32)

        # profile core
        k[1, 1, :, 0] = 1

        # + pattern
        k[0, 1, :, 1] = 1
        k[1, 0, :, 1] = 1
        k[2, 1, :, 1] = 1
        k[1, 2, :, 1] = 1

        # x pattern
        k[0, 2, :, 2] = 1
        k[2, 0, :, 2] = 1
        k[0, 0, :, 2] = 1
        k[2, 2, :, 2] = 1

        self.kernel = tf.constant(np.repeat(k, input_channels, axis=2))

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(
            inputs, self.kernel, strides=[1] * 4, padding="VALID"
        )


class SymmetricDepthwiseConv2D(layers.Layer):
    def __init__(self, depth_multiplier=1, input_channels=6, **kwargs):
        super().__init__()

        self.input_channels = input_channels
        self.depth_multiplier = depth_multiplier

        self.pooling = SymmetricPooling(input_channels=self.input_channels)
        self.dense_layers = []
        for _ in range(self.input_channels):
            self.dense_layers.append(layers.Dense(depth_multiplier))

    def call(self, inputs):
        pooled_inputs = self.pooling(inputs)
        pooled_inputs_by_layer = tf.split(pooled_inputs, self.input_channels, axis=-1)
        pooled_inputs_by_layer = [
            dense_layer(x)
            for dense_layer, x in zip(self.dense_layers, pooled_inputs_by_layer)
        ]
        outputs = layers.Concatenate()(pooled_inputs_by_layer)
        return outputs

    def get_config(self):
        base_config = super().get_config()
        config = {
            "depth_multiplier": self.depth_multiplier,
            "dense_layers": keras.saving.serialize_keras_object(self.dense_layers),
            "input_channels": self.input_channels
        }
        return {**base_config, **config}


class QSymmetricDepthwiseConv2D(SymmetricDepthwiseConv2D):
    def __init__(
        self,
        kernel_quantizer,
        bias_quantizer,
        depth_multiplier=1,
        input_channels=6,
        **kwargs
    ):
        super().__init__(
            depth_multiplier=depth_multiplier, input_channels=input_channels, **kwargs
        )

        self.bias_quantizer = bias_quantizer
        self.kernel_quantizer = kernel_quantizer
        self.dense_layers = []
        for _ in range(input_channels):
            self.dense_layers.append(
                QDense(
                    depth_multiplier,
                    kernel_quantizer=kernel_quantizer,
                    bias_quantizer=bias_quantizer,
                )
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "bias_quantizer": keras.saving.serialize_keras_object(self.bias_quantizer),
            "kernel_quantizer": keras.saving.serialize_keras_object(
                self.kernel_quantizer
            ),
        }
        return {**base_config, **config}
    