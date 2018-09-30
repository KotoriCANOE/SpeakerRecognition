import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = layers.Swish
DATA_FORMAT = 'NCHW'

# Discriminator

class DiscriminatorConfig:
    def __init__(self):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 1
        self.num_labels = None
        # model parameters
        self.activation = ACTIVATION
        self.normalization = 'Batch'
        self.embed_size = 512
        # train parameters
        self.random_seed = 0
        self.var_ema = 0.999
        self.weight_decay = 1e-6
        self.dropout = 0

class Discriminator(DiscriminatorConfig):
    def __init__(self, name='Discriminator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    def ResBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], format=DATA_FORMAT,
        activation=None, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, normalizer, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def EBlock(self, last, channels, resblocks=1,
        kernel=[1, 4], stride=[1, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # residual blocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # dense connection
        with tf.variable_scope('DenseConnection'):
            last = layers.SEUnit(last, channels, format, collections)
            if stride != 1 or stride != [1, 1]:
                pool_stride = [1, 1] + stride if format == 'NCHW' else [1] + stride + [1]
                skip = tf.nn.avg_pool(skip, pool_stride, pool_stride, 'SAME', format)
            last = tf.concat([skip, last], -3 if format == 'NCHW' else -1)
        return last

    def __call__(self, last, reuse=None):
        # parameters
        format = self.data_format
        kernel1 = [1, 4]
        stride1 = [1, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, [1, 7], [1, 1],
                    format, None, None, regularizer)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 32, 0, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 32, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                last = self.EBlock(last, 40, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                last = self.EBlock(last, 40, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_5'):
                last = self.EBlock(last, 48, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_6'):
                last = self.EBlock(last, 48, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_7'):
                last = self.EBlock(last, 56, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_8'):
                last = self.EBlock(last, 56, 3, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_9'):
                last = self.EBlock(last, 64, 3, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_10'):
                last = self.EBlock(last, 64, 3, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('GlobalAveragePooling'):
                last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
            with tf.variable_scope('FCBlock'):
                skip = last
                last_channels = last.shape.as_list()[-1]
                last = slim.fully_connected(last, last_channels, activation, None,
                    weights_regularizer=regularizer)
                last = slim.fully_connected(last, self.embed_size, None, None,
                    weights_regularizer=regularizer)
                if self.embed_size == last_channels:
                    last += skip
                embeddings = last
            with tf.variable_scope('OutBlock'):
                if self.dropout > 0:
                    last = tf.layers.dropout(last, self.dropout, training=self.training)
                last = slim.fully_connected(last, self.num_labels, None, None,
                    weights_regularizer=regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('EMA'):
                self.rvars = {**{self.ema.average_name(var): var for var in self.tvars},
                    **{var.op.name: var for var in self.mvars}}
        return embeddings, last

    def apply_ema(self, update_ops=[]):
        if not self.var_ema:
            return update_ops
        with tf.variable_scope('EMA'):
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.tvars)]
            self.svars = [self.ema.average(var) for var in self.tvars] + self.mvars
        return update_ops

class Discriminator2(DiscriminatorConfig):
    def __init__(self, name='Discriminator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    def ResBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], format=DATA_FORMAT,
        activation=None, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, normalizer, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def EBlock(self, last, l=4, k=32, channels=None, bottleneck=False,
        kernel=[1, 3], stride=[1, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        channel_index = -3 if format == 'NCHW' else -1
        last_channels = last.shape.as_list()[channel_index]
        # dense layers
        for _ in range(l):
            skip = last
            # bottleneck
            if bottleneck:
                if normalizer: last = normalizer(last)
                if activation: last = activation(last)
                last = slim.conv2d(last, k, [1, 1], [1, 1], 'SAME', format,
                    1, None, None, weights_initializer=initializer,
                    weights_regularizer=regularizer, variables_collections=collections)
            # pre-activation
            if normalizer: last = normalizer(last)
            if activation: last = activation(last)
            # convolution
            last = slim.conv2d(last, k, kernel, [1, 1], 'SAME', format,
                1, None, None, weights_initializer=initializer,
                weights_regularizer=regularizer, variables_collections=collections)
            # squeeze-and-excitation
            last = layers.SEUnit(last, channels, format, collections)
            # concatenate
            last = tf.concat([skip, last], channel_index)
            last_channels += k
        # compression
        if channels and channels != last_channels:
            last = slim.conv2d(last, channels, [1, 1], [1, 1], 'SAME', format,
                1, None, None, weights_initializer=initializer,
                weights_regularizer=regularizer, variables_collections=collections)
            last_channels = channels
        # pooling
        strides = [1, 1] + stride if format == 'NCHW' else [1] + stride + [1]
        last = tf.nn.avg_pool(last, strides, strides, 'SAME', format)
        # return
        return last

    def __call__(self, last, reuse=None):
        # parameters
        format = self.data_format
        kernel1 = [1, 7]
        stride1 = [1, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, [1, 7], [1, 1],
                    format, None, None, regularizer)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 4, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 6, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                last = self.EBlock(last, 8, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                last = self.EBlock(last, 12, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_5'):
                last = self.EBlock(last, 16, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_6'):
                last = self.EBlock(last, 16, 16, None, False, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('GlobalAveragePooling'):
                last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
            with tf.variable_scope('FCBlock'):
                skip = last
                last_channels = last.shape.as_list()[-1]
                last = slim.fully_connected(last, last_channels, activation, None,
                    weights_regularizer=regularizer)
                last = slim.fully_connected(last, self.embed_size, None, None,
                    weights_regularizer=regularizer)
                if self.embed_size == last_channels:
                    last += skip
                embeddings = last
            with tf.variable_scope('OutBlock'):
                if self.dropout > 0:
                    last = tf.layers.dropout(last, self.dropout, training=self.training)
                last = slim.fully_connected(last, self.num_labels, None, None,
                    weights_regularizer=regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('EMA'):
                self.rvars = {**{self.ema.average_name(var): var for var in self.tvars},
                    **{var.op.name: var for var in self.mvars}}
        return embeddings, last

    def apply_ema(self, update_ops=[]):
        if not self.var_ema:
            return update_ops
        with tf.variable_scope('EMA'):
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.tvars)]
            self.svars = [self.ema.average(var) for var in self.tvars] + self.mvars
        return update_ops
