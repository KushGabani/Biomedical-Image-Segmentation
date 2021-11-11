import tensorflow as tf


class DoubleConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filter, layer_name):
        super(DoubleConvolutionBlock, self).__init__(name="double_conv_3x3_" + layer_name)
        self.filter = filter
        self.pool = tf.keras.layers.MaxPool2D((2, 2))
        self.layers = [
            tf.keras.layers.Conv2D(
                filter,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            ),
            tf.keras.layers.Conv2D(
                filter,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False
            )
        ]

    def call(self, inputs, training=None, pool=True):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            if training:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        skip = x

        return self.pool(x) if pool else x, skip


class UpSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filter, layer_name):
        super(UpSamplingBlock, self).__init__(name="up_sampling_"+layer_name)
        self.filter = filter
        self.upsample_convolution = [
            tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear"),
            tf.keras.layers.Conv2D(
                self.filter,
                (2, 2),
                padding="same",
                kernel_initializer="he_normal",
                activation="relu"
            ),
        ]
        self.double_conv_3x3 = DoubleConvolutionBlock(self.filter, layer_name)

    def call(self, inputs, skip=None):
        x = inputs
        for layer in self.upsample_convolution:
            x = layer(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate()([x, skip])

        x, _ = self.double_conv_3x3.call(x, pool=False)
        return x


class UNet(tf.keras.models.Model):
    def __init__(self, n_classes=1, filters=None, end_activation="sigmoid"):
        super(UNet, self).__init__(name="UNet for semantic segmentation")
        self.filters = filters
        if not self.filters:
            self.filters = [64, 128, 256, 512]
        self.end_activation = end_activation
        self.n_classes = n_classes

        self.contractions = []
        for filter in self.filters:
            self.contractions.append(DoubleConvolutionBlock(filter, layer_name=str(filter)))

        self.latent_space = DoubleConvolutionBlock(self.filters[-1]*2, layer_name="latent_{}".format(self.filters[-1]*2))

        self.expansions = []
        for filter in self.filters[::-1]:
            self.expansions.append(UpSamplingBlock(filter, layer_name=str(filter)))

        self.conv_1x1 = tf.keras.layers.Conv2D(
            self.n_classes,
            (1, 1),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            activation=end_activation
        )

    def call(self, inputs):
        x = inputs
        skip_connections = []
        for contraction in self.contractions:
            x, skip = contraction.call(x)
            skip_connections.append(skip)

        x, _ = self.latent_space.call(x, pool=False)

        for expansion, skip_connection in zip(self.expansions, skip_connections[::-1]):
            x = expansion.call(x, skip_connection)

        x = self.conv_1x1(x)
        return x
