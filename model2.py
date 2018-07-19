import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = layers.Swish
DATA_FORMAT = 'NCHW'

class SRN:
    def __init__(self, config=None):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 1
        self.out_channels = 256
        self.batch_size = None
        self.patch_height = None
        self.patch_width = None
        # model parameters
        self.embed_size = 512
        self.batch_norm = 0.999
        # train parameters
        self.random_seed = None
        self.dropout = 0
        self.var_ema = 0.999
        # loss parameters
        self.triplet_margin = 1.0
        # generator parameters
        self.generator_acti = ACTIVATION
        self.generator_wd = 1e-6
        self.generator_lr = 1e-3
        self.generator_lr_step = 1000
        self.generator_vkey = 'generator_var'
        self.generator_lkey = 'generator_loss'
        # collections
        self.train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None] * 4
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.label_shape = [None]
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    @staticmethod
    def add_arguments(argp):
        # model parameters
        argp.add_argument('--embed-size', type=int, default=512)
        argp.add_argument('--batch-norm', type=float, default=0.999)
        # training parameters
        argp.add_argument('--dropout', type=float, default=0)
        argp.add_argument('--var-ema', type=float, default=0.999)
        argp.add_argument('--generator-wd', type=float, default=1e-6)
        argp.add_argument('--generator-lr', type=float, default=1e-3)
        argp.add_argument('--generator-lr-step', type=int, default=1000)
        # loss parameters
        argp.add_argument('--triplet-margin', type=float, default=0.5)

    def ResBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # skip connection
        last = layers.SEUnit(last, channels, format, collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def EBlock(self, last, channels, resblocks=1, kernel=[1, 4], stride=[1, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        in_channels = skip.get_shape()[-3 if format == 'NCHW' else -1]
        if activation: last = activation(last)
        if in_channels > channels:
            last = slim.conv2d(last, channels,
                [1, 1], [1, 1], 'SAME', format,
                1, activation, None, weights_initializer=initializer,
                weights_regularizer=regularizer, variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        with tf.variable_scope('DenseConnection'):
            last = layers.SEUnit(last, channels, format, collections)
            if stride != 1 or stride != [1, 1]:
                pool_stride = [1, 1] + stride if format == 'NCHW' else [1] + stride + [1]
                skip = tf.nn.avg_pool(skip, pool_stride, pool_stride, 'SAME', format)
            last = tf.concat([skip, last], -3 if format == 'NCHW' else -1)
        return last

    def generator(self, last):
        # parameters
        main_scope = 'generator'
        format = self.data_format
        var_key = self.generator_vkey
        kernel1 = [1, 3]
        stride1 = [1, 2]
        # model definition
        with tf.variable_scope(main_scope):
            # states
            self.g_training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # function objects
            activation = self.generator_acti
            if self.batch_norm > 0:
                normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=False,
                    is_training=self.g_training, data_format=format, renorm=False)
            else:
                normalizer = None
            regularizer = slim.l2_regularizer(self.generator_wd)
            # network
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, [1, 7], [1, 1], format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 32, 0, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 32, 1, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_3'):
                last = self.EBlock(last, 32, 1, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_4'):
                last = self.EBlock(last, 48, 2, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_5'):
                last = self.EBlock(last, 48, 2, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_6'):
                last = self.EBlock(last, 48, 2, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_7'):
                last = self.EBlock(last, 48, 3, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_8'):
                last = self.EBlock(last, 64, 3, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_9'):
                last = self.EBlock(last, 64, 3, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('EBlock_10'):
                last = self.EBlock(last, 64, 3, kernel1, stride1, format, activation,
                    normalizer, regularizer, var_key)
            with tf.variable_scope('GlobalAveragePooling'):
                last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
            with tf.variable_scope('FCBlock'):
                skip = last
                last_channels = last.shape.as_list()[-1]
                last = slim.fully_connected(last, last_channels, activation, None,
                    weights_regularizer=regularizer, variables_collections=var_key)
                last = slim.fully_connected(last, self.embed_size, None, None,
                    weights_regularizer=regularizer, variables_collections=var_key)
                if self.embed_size == last_channels:
                    last += skip
                self.embeddings = last
            with tf.variable_scope('OutBlock'):
                if self.dropout > 0:
                    last = tf.layers.dropout(last, self.dropout, training=self.g_training)
                last = slim.fully_connected(last, self.out_channels, None, None,
                    weights_regularizer=regularizer, variables_collections=var_key)
        # trainable/model/save/restore variables
        self.g_tvars = tf.trainable_variables(main_scope)
        self.g_mvars = tf.model_variables(main_scope)
        self.g_mvars = [i for i in self.g_mvars if i not in self.g_tvars]
        self.g_svars = list(set(self.g_tvars + self.g_mvars))
        self.g_rvars = self.g_svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('variables_ema'):
                self.g_rvars = {**{self.ema.average_name(var): var for var in self.g_tvars},
                    **{var.op.name: var for var in self.g_mvars}}
        return last

    def build_g_loss(self, labels, outputs, embeddings):
        self.g_log_losses = []
        update_ops = []
        loss_key = self.generator_lkey
        with tf.variable_scope(loss_key):
            # softmax cross entropy
            onehot_labels = tf.one_hot(labels, self.out_channels)
            cross_loss = tf.losses.softmax_cross_entropy(onehot_labels, outputs, 1.0)
            update_ops.append(self.loss_summary('cross_loss', cross_loss, self.g_log_losses))
            # accuracy
            accuracy = tf.contrib.metrics.accuracy(labels, tf.argmax(outputs, -1))
            update_ops.append(self.loss_summary('accuracy', accuracy, self.g_log_losses))
            # center loss
            from center_loss import get_center_loss
            lambda_ = 0.003
            center_loss, centers, centers_update_op = get_center_loss(embeddings, labels, self.out_channels)
            tf.losses.add_loss(center_loss * lambda_)
            update_ops.append(centers_update_op)
            update_ops.append(self.loss_summary('center_loss', center_loss, self.g_log_losses))
            update_ops.append(self.loss_summary('fraction_positive_triplets', 0, self.g_log_losses))
            '''
            # triplet loss
            from triplet_loss import batch_all, batch_hard
            triplet_loss, fraction = batch_all(labels, embeddings, self.triplet_margin)
            # triplet_loss, fraction = batch_hard(labels, embeddings, self.triplet_margin)
            tf.losses.add_loss(triplet_loss)
            update_ops.append(self.loss_summary('triplet_loss', triplet_loss, self.g_log_losses))
            update_ops.append(self.loss_summary('fraction_positive_triplets', fraction, self.g_log_losses))
            '''
            # total loss
            losses = tf.losses.get_losses(loss_key)
            g_main_loss = tf.add_n(losses, 'g_main_loss')
            # regularization loss
            g_reg_losses = tf.losses.get_regularization_losses('generator')
            g_reg_loss = tf.add_n(g_reg_losses)
            update_ops.append(self.loss_summary('g_reg_loss', g_reg_loss))
            # final loss
            self.g_loss = g_main_loss + g_reg_loss
            update_ops.append(self.loss_summary('g_loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('g_losses_accumulator')

    def build_model(self, inputs=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        # forward pass
        self.outputs = self.generator(self.inputs)
        # embeddings
        self.embeddings = tf.identity(self.embeddings, name='Embedding')
        # outputs
        self.outputs = tf.identity(self.outputs, name='Output')
        # all the restore variables
        self.rvars = self.g_rvars
        # return outputs
        return self.outputs

    def build_train(self, inputs=None, labels=None):
        # reference outputs
        if labels is None:
            self.labels = tf.placeholder(tf.int64, self.label_shape, name='Label')
        else:
            self.labels = tf.identity(labels, name='Label')
            self.labels.set_shape(self.label_shape)
        # build model
        self.build_model(inputs)
        # build generator loss
        self.build_g_loss(self.labels, self.outputs, self.embeddings)
        # return total loss
        return self.g_loss

    def train(self, global_step):
        # saving memory with gradient checkpoints
        #self.set_reuse_checkpoints()
        #g_ckpt = tf.get_collection('checkpoints', 'generator')
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # learning rate
        g_lr = tf.train.cosine_decay_restarts(self.generator_lr,
            global_step, self.generator_lr_step)
        self.train_sums.append(tf.summary.scalar('generator_lr', g_lr))
        # optimizer
        g_opt = tf.contrib.opt.NadamOptimizer(g_lr)
        with tf.control_dependencies(update_ops):
            #g_grads_vars = self.compute_gradients(self.g_loss, self.g_tvars, g_ckpt)
            g_grads_vars = g_opt.compute_gradients(self.g_loss, self.g_tvars)
            update_ops = [g_opt.apply_gradients(g_grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in g_grads_vars:
            self.train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        if self.var_ema > 0:
            with tf.variable_scope('variables_ema'):
                with tf.control_dependencies(update_ops):
                    update_ops = [self.ema.apply(self.g_tvars)]
                self.g_svars = [self.ema.average(var) for var in self.g_tvars] + self.g_mvars
        # all the saver variables
        self.svars = self.g_svars
        # return training op
        with tf.control_dependencies(update_ops):
            g_train_op = tf.no_op('g_train')
        return g_train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('loss_summary/' + name):
            # internal variables
            loss_sum = tf.get_variable('sum', (), tf.float32, tf.initializers.zeros(tf.float32))
            loss_count = tf.get_variable('count', (), tf.float32, tf.initializers.zeros(tf.float32))
            # accumulate to sum and count
            acc_sum = loss_sum.assign_add(loss, True)
            acc_count = loss_count.assign_add(1.0, True)
            # calculate mean
            loss_mean = tf.divide(loss_sum, loss_count, 'mean')
            if collection is not None:
                collection.append(loss_mean)
            # reset sum and count
            with tf.control_dependencies([loss_mean]):
                clear_sum = loss_sum.assign(0.0, True)
                clear_count = loss_count.assign(0.0, True)
            # log summary
            with tf.control_dependencies([clear_sum, clear_count]):
                self.loss_sums.append(tf.summary.scalar(name, loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        train_summary = tf.summary.merge(self.train_sums) if self.train_sums else None
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        return train_summary, loss_summary

    @staticmethod
    def set_reuse_checkpoints():
        import re
        # https://stackoverflow.com/a/36893840
        # get the name of all the tensors output by weight operations
        # MatMul, Conv2D, etc.
        graph = tf.get_default_graph()
        nodes = graph.as_graph_def().node
        regex = re.compile(r'^.+(?:MatMul|Conv2D|conv2d_transpose|BiasAdd)$')
        op_names = [n.name for n in nodes if re.match(regex, n.name)]
        tensors = [graph.get_tensor_by_name(n + ':0') for n in op_names]
        # add these tensors to collection 'checkpoints'
        for tensor in tensors:
            tf.add_to_collection('checkpoints', tensor)

    @staticmethod
    def compute_gradients(loss, var_list, checkpoints='collection'):
        # https://github.com/openai/gradient-checkpointing
        from memory_saving_gradients import gradients
        grads = gradients(loss, var_list, checkpoints=checkpoints)
        grads_vars = list(zip(grads, var_list))
        return grads_vars
