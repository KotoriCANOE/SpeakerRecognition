import tensorflow as tf
from network import Discriminator

DATA_FORMAT = 'NCHW'

class Model:
    def __init__(self, config=None):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 1
        self.num_labels = None
        # loss parameters
        self.triplet_margin = None
        # discriminator parameters
        self.d_lr = None
        self.d_lr_step = None
        # collections
        self.d_train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        self.config = config
        if config is not None:
            self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None, None, None, None]
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.label_shape = [None]

    @staticmethod
    def add_arguments(argp):
        # model parameters
        argp.add_argument('--embed-size', type=int, default=512)
        # training parameters
        argp.add_argument('--dropout', type=float, default=0)
        argp.add_argument('--var-ema', type=float, default=0.999)
        argp.add_argument('--d-lr', type=float, default=1e-3)
        argp.add_argument('--d-lr-step', type=int, default=1000)
        # loss parameters
        argp.add_argument('--triplet-margin', type=float, default=0.5)
        argp.add_argument('--center-decay', type=float, default=0.95)

    def build_model(self, inputs=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        # forward pass
        self.discriminator = Discriminator('Discriminator', self.config)
        self.outputs = self.discriminator(self.inputs, reuse=None)
        # embeddings
        self.embeddings = tf.identity(self.embeddings, name='Embedding')
        # outputs
        self.outputs = tf.identity(self.outputs, name='Output')
        # all the saver variables
        self.svars = self.discriminator.svars
        # all the restore variables
        self.rvars = self.discriminator.rvars
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
        # build discriminator loss
        self.build_d_loss(self.labels, self.outputs, self.embeddings)

    def build_d_loss(self, labels, outputs, embeddings):
        self.d_log_losses = []
        update_ops = []
        loss_key = 'DiscriminatorLoss'
        with tf.variable_scope(loss_key):
            # softmax cross entropy
            onehot_labels = tf.one_hot(labels, self.num_labels)
            cross_loss = tf.losses.softmax_cross_entropy(onehot_labels, outputs, 1.0)
            update_ops.append(self.loss_summary('cross_loss', cross_loss, self.d_log_losses))
            # accuracy
            accuracy = tf.contrib.metrics.accuracy(labels, tf.argmax(outputs, -1))
            update_ops.append(self.loss_summary('accuracy', accuracy, self.d_log_losses))
            '''
            # center loss
            from center_loss import get_center_loss_unbias
            lambda_center = 0.003
            center_loss, centers, centers_update_ops = get_center_loss_unbias(
                embeddings, labels, self.num_labels, self.center_decay)
            center_loss *= lambda_center
            tf.losses.add_loss(center_loss)
            update_ops.extend(centers_update_ops)
            update_ops.append(self.loss_summary('center_loss', center_loss, self.d_log_losses))
            update_ops.append(self.loss_summary('fraction_positive_triplets', 0, self.d_log_losses))
            '''
            # triplet loss
            from triplet_loss import batch_all
            triplet_loss, fraction = batch_all(labels, embeddings, self.triplet_margin)
            tf.losses.add_loss(triplet_loss)
            update_ops.append(self.loss_summary('triplet_loss', triplet_loss, self.d_log_losses))
            update_ops.append(self.loss_summary('fraction_positive_triplets', fraction, self.d_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            main_loss = tf.add_n(losses, 'main_loss')
            # regularization loss
            reg_losses = tf.losses.get_regularization_losses('discriminator')
            reg_loss = tf.add_n(reg_losses)
            update_ops.append(self.loss_summary('reg_loss', reg_loss))
            # final loss
            self.d_loss = main_loss + reg_loss
            update_ops.append(self.loss_summary('loss', self.d_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.d_losses_acc = tf.no_op('accumulator')

    def train_d(self, global_step):
        model = self.discriminator
        # saving memory with gradient checkpoints
        # self.set_reuse_checkpoints()
        # ckpt = tf.get_collection('checkpoints', 'Discriminator')
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Discriminator')
        # learning rate
        lr = tf.train.cosine_decay_restarts(self.d_lr,
            global_step, self.d_lr_step, t_mul=2.0, m_mul=1.0, alpha=1e-1)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.999)
        self.d_train_sums.append(tf.summary.scalar('Discriminator/LR', lr))
        # optimizer
        opt = tf.contrib.opt.NadamOptimizer(lr)
        with tf.control_dependencies(update_ops):
            # grads_vars = self.compute_gradients(self.d_loss, model.tvars, ckpt)
            grads_vars = opt.compute_gradients(self.d_loss, model.tvars)
            update_ops = [opt.apply_gradients(grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.d_train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.d_train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        update_ops = model.apply_ema(update_ops)
        # all the saver variables
        self.svars = model.svars
        # return training op
        with tf.control_dependencies(update_ops):
            train_op = tf.no_op('train_d')
        return train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('LossSummary/' + name):
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
                self.loss_sums.append(tf.summary.scalar('value', loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        d_train_summary = tf.summary.merge(self.d_train_sums) if self.d_train_sums else None
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        return d_train_summary, loss_summary

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
