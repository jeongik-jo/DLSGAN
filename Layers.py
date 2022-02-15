import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp


class EqDense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True):
        super(EqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]), name=self.name + '_w')
        self.he_std = tf.sqrt(2.0 / tf.cast(input_shape[-1], dtype='float32'))

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_vector = tf.matmul(inputs, self.w) * self.he_std
        if self.use_bias:
            return self.activation(feature_vector + self.b)
        else:
            return self.activation(feature_vector)


class Blur(kr.layers.Layer):
    def __init__(self, upscale=False, downscale=False, padding=None):
        super(Blur, self).__init__()
        self.upscale = upscale
        self.downscale = downscale
        self.padding = padding

        assert (upscale and downscale) != True

    def build(self, input_shape):
        kernel = tf.convert_to_tensor([1, 3, 3, 1.0])
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)

        if self.upscale:
            kernel = kernel * 4.0
            self.reshape_layer = kr.layers.Reshape([input_shape[1], input_shape[2] * 2, input_shape[3] * 2])
            self.padding = [[0, 0], [0, 0], [2, 1], [2, 1]]
        elif self.downscale:
            self.padding = [[0, 0], [0, 0], [1, 1], [1, 1]]
        else:
            self.padding = [[0, 0], [0, 0]] + self.padding

        self.kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[1], 1])

    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            feature_maps = tf.stack([inputs, tf.zeros_like(inputs)], axis=3)
            feature_maps = tf.stack([feature_maps, tf.zeros_like(feature_maps)], axis=5)
            feature_maps = self.reshape_layer(feature_maps)
            return tf.nn.depthwise_conv2d(input=feature_maps, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 2, 2],
                                          padding=self.padding, data_format='NCHW')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')


class EqConv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True):
        super(EqConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_filters = input_shape[1]

        self.he_std = tf.sqrt(2.0 / tf.cast(self.kernel_size * self.kernel_size * input_filters, dtype='float32'))
        self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                             name=self.name + '_w')

        padding = self.kernel_size // 2
        self.padding = [[0, 0], [0, 0], [padding, padding], [padding, padding]]

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.filters, 1, 1]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_maps = tf.nn.conv2d(inputs, self.w, strides=1, padding=self.padding, data_format='NCHW') * self.he_std

        if self.use_bias:
            feature_maps = self.activation(feature_maps + self.b)
        else:
            feature_maps = self.activation(feature_maps)

        return feature_maps


class Fir(kr.layers.Layer):
    def __init__(self, kernel, padding=None, upscale=False, downscale=False):
        super(Fir, self).__init__()
        self.kernel = kernel
        self.padding = padding
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, input_shape):
        if self.upscale:
            self.reshape_layer = kr.layers.Reshape([input_shape[1], input_shape[2] * 2, input_shape[3] * 2])
        if self.padding == None:
            padding_0 = (self.kernel.shape[0] - 1) // 2
            padding_1 = self.kernel.shape[0] - 1 - padding_0
            self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]
        else:
            self.padding = [[0, 0], [0, 0]] + self.padding

        self.kernel = tf.tile(self.kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[1], 1])

    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            feature_maps = tf.stack([inputs, tf.zeros_like(inputs)], axis=3)
            feature_maps = tf.stack([feature_maps, tf.zeros_like(feature_maps)], axis=5)
            feature_maps = self.reshape_layer(feature_maps)
            return tf.nn.depthwise_conv2d(input=feature_maps, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 2, 2],
                                          padding=self.padding, data_format='NCHW')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')


def get_haar_kernels():
    haar_l = tf.convert_to_tensor([[1.0 / tf.sqrt(2.0), 1.0 / tf.sqrt(2.0)]])
    haar_h = tf.convert_to_tensor([[-1.0 / tf.sqrt(2.0), 1.0 / tf.sqrt(2.0)]])
    haar_l_t = tf.transpose(haar_l, perm=[1, 0])
    haar_h_t = tf.transpose(haar_h, perm=[1, 0])

    haar_ll = haar_l_t * haar_l
    haar_lh = haar_h_t * haar_l
    haar_hl = haar_l_t * haar_h
    haar_hh = haar_h_t * haar_h

    return [haar_ll, haar_lh, haar_hl, haar_hh]


class HaarTransform(kr.layers.Layer):
    def __init__(self):
        super(HaarTransform, self).__init__()
        self.fir_layers = [Fir(kernel=kernel, downscale=True) for kernel in get_haar_kernels()]

    def call(self, inputs, *args, **kwargs):
        return tf.concat([fir_layer(inputs) for fir_layer in self.fir_layers], axis=1)


class InverseHaarTransform(kr.layers.Layer):
    def __init__(self):
        super(InverseHaarTransform, self).__init__()
        kernels = get_haar_kernels()
        kernels[1] = -kernels[1]
        kernels[2] = -kernels[2]

        self.fir_layers = [Fir(kernel=kernel, padding=[[1, 0], [1, 0]], upscale=True) for kernel in kernels]

    def call(self, inputs, *args, **kwargs):
        return tf.math.add_n([fir_layer(feature_maps) for feature_maps, fir_layer
                              in zip(tf.split(inputs, 4, axis=1), self.fir_layers)])


class To4RGB(kr.layers.Layer):
    def __init__(self):
        super(To4RGB, self).__init__()

    def build(self, input_shape):
        self.iwt = InverseHaarTransform()
        self.upscale = Blur(upscale=True)
        self.dwt = HaarTransform()
        self.conv_layer = EqConv2D(filters=3 * 4, kernel_size=1)

    def call(self, inputs, *args, **kwargs):
        image = inputs[0]
        feature_maps = inputs[1]

        return (self.dwt(self.upscale(self.iwt(image))) + self.conv_layer(feature_maps)) / tf.sqrt(2.0)


class From4RGB(kr.layers.Layer):
    def __init__(self):
        super(From4RGB, self).__init__()

    def build(self, input_shape):
        self.conv_layer = EqConv2D(filters=input_shape[1][1], kernel_size=3, activation=tf.nn.leaky_relu)
        self.iwt = InverseHaarTransform()
        self.downsample = Blur(downscale=True)
        self.dwt = HaarTransform()

    def call(self, inputs, *args, **kwargs):
        image = inputs[0]
        feature_maps = inputs[1]

        image = self.dwt(self.downsample(self.iwt(image)))

        feature_maps = (self.conv_layer(image) + feature_maps) / tf.sqrt(2.0)
        return image, feature_maps


class Generator(kr.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

    def build(self, input_shape):
        latent_vector = kr.Input([hp.latent_vector_dim])
        feature_vector = EqDense(units=1024, activation=tf.nn.leaky_relu)(latent_vector)
        feature_vector = EqDense(units=1024*4*4, activation=tf.nn.leaky_relu)(feature_vector)
        feature_maps = kr.layers.Reshape([1024, 4, 4])(feature_vector)

        fake_image = EqConv2D(filters=3 * 4, kernel_size=1)(feature_maps)

        filters_sizes = [512, 512, 512, 256, 128, 64]
        for filters in filters_sizes:
            feature_maps = Blur(upscale=True)(feature_maps)
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            fake_image = To4RGB()([fake_image, feature_maps])

        fake_image = tf.transpose(InverseHaarTransform()(fake_image), [0, 2, 3, 1])
        self.model = kr.Model(latent_vector, fake_image)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class Discriminator(kr.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

    def build(self, input_shape):
        input_image = kr.Input(shape=[hp.image_resolution, hp.image_resolution, 3])
        feature_maps = image = HaarTransform()(tf.transpose(input_image, [0, 3, 1, 2]))

        filters_sizes = [64, 128, 256, 512, 512, 512]
        for filters in filters_sizes:
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = EqConv2D(filters=tf.math.minimum(filters * 2, 512), kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = Blur(downscale=True)(feature_maps)
            image, feature_maps = From4RGB()([image, feature_maps])

        feature_maps = EqConv2D(filters=1024, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
        feature_vector = kr.layers.Flatten()(feature_maps)
        feature_vector = EqDense(units=1024, activation=tf.nn.leaky_relu)(feature_vector)
        adv_value = EqDense(units=1)(feature_vector)
        latent_vector = EqDense(units=hp.latent_vector_dim)(feature_vector)

        self.model = kr.Model(input_image, [adv_value, latent_vector])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)

