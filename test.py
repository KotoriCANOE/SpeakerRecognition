import tensorflow as tf
import numpy as np
import os
from utils import eprint, reset_random, create_session
from data import Data
from model2 import SRN

class SRNTest(SRN):
    def test_loss(self, labels, outputs, embeddings):
        self.log_losses = []
        update_ops = []
        loss_key = 'test_loss'
        with tf.variable_scope(loss_key):
            # softmax cross entropy
            onehot_labels = tf.one_hot(labels, self.out_channels)
            cross_loss = tf.losses.softmax_cross_entropy(onehot_labels, outputs, 1.0)
            update_ops.append(self.loss_summary('cross_loss', cross_loss, self.log_losses))
            # accuracy
            accuracy = tf.contrib.metrics.accuracy(labels, tf.argmax(outputs, -1))
            update_ops.append(self.loss_summary('accuracy', accuracy, self.log_losses))
            # triplet loss
            from triplet_loss import batch_all, batch_hard
            triplet_loss, fraction = batch_all(labels, embeddings, self.triplet_margin)
            # triplet_loss, fraction = batch_hard(labels, embeddings, self.triplet_margin)
            tf.losses.add_loss(triplet_loss)
            update_ops.append(self.loss_summary('triplet_loss', triplet_loss, self.log_losses))
            update_ops.append(self.loss_summary('fraction_positive_triplets', fraction, self.log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            total_loss = tf.add_n(losses, 'total_loss')
            update_ops.append(self.loss_summary('total_loss', total_loss, self.log_losses))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.losses_acc = tf.no_op('losses_accumulator')

    def build_test(self, inputs=None, labels=None):
        # reference outputs
        if labels is None:
            self.labels = tf.placeholder(tf.int64, self.label_shape, name='Label')
        else:
            self.labels = tf.identity(labels, name='Label')
            self.labels.set_shape(self.label_shape)
        # build model
        self.build_model(inputs)
        # build generator loss
        self.test_loss(self.labels, self.outputs, self.embeddings)

# class for testing session
class Test:
    def __init__(self, config):
        self.random_seed = None
        self.device = None
        self.postfix = None
        self.train_dir = None
        self.test_dir = None
        self.model_file = None
        self.log_file = None
        self.batch_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)

    def initialize(self):
        import sys
        # arXiv 1509.09308
        # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        # create testing directory
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError('Could not find folder {}'.format(self.train_dir))
            sys.exit()
        if os.path.exists(self.test_dir):
            eprint('Confirm removing {}\n[Y/n]'.format(self.test_dir))
            if input() != 'Y':
                import sys
                sys.exit()
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
            eprint('Removed: ' + self.test_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        self.data = Data(self.config)
        self.epoch_steps = self.data.epoch_steps
        self.max_steps = self.data.max_steps
        # pre-computing testing set
        self.test_inputs = []
        self.test_labels = []
        data_gen = self.data.gen_main()
        for _inputs, _labels in data_gen:
            self.test_inputs.append(_inputs)
            self.test_labels.append(_labels)

    def build_graph(self):
        with tf.device(self.device):
            self.model = SRNTest(self.config)
            self.model.build_test()
            _, self.loss_summary = self.model.get_summaries()

    def build_saver(self):
        # a Saver object to restore the variables with mappings
        self.saver = tf.train.Saver(self.model.rvars)

    def run_last(self, sess):
        # initialize all variables
        initializers = (tf.initializers.global_variables(),
            tf.initializers.local_variables())
        sess.run(initializers)
        # latest checkpoint or specific model
        if self.model_file is None:
            ckpt = tf.train.latest_checkpoint(self.train_dir)
        else:
            ckpt = os.path.join(self.train_dir, self.model_file)
        self.saver.restore(sess, ckpt)
        # to be fetched
        fetch = [self.model.losses_acc, self.model.embeddings, self.model.outputs]
        embeddings = []
        outputs = []
        # loop over batches
        for step in range(self.epoch_steps):
            feed_dict = {'Input:0': self.test_inputs[step],
                'Label:0': self.test_labels[step]}
            _, _embeddings, _outputs = sess.run(fetch, feed_dict)
            embeddings.append(_embeddings)
            outputs.append(_outputs)
        # get summaries
        fetch = [self.loss_summary] + self.model.log_losses
        test_ret = sess.run(fetch)
        # log result
        if self.log_file:
            from datetime import datetime
            last_log = ('cross: {:.5}, accuracy: {:.5}'
                ', triplet: {:.5}, fraction: {:.5}; total: {:.5}'
                .format(*test_ret[1:]))
            with open(self.log_file, 'a', encoding='utf-8') as fd:
                fd.write('Testing No.{}\n'.format(self.postfix))
                fd.write(self.test_dir + '\n')
                fd.write('{}\n'.format(datetime.now()))
                fd.write(last_log + '\n\n')
        # write embeddings
        labels = np.concatenate(self.test_labels, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)
        with open(os.path.join(self.test_dir, 'embeddings.npz'), 'wb') as fd:
            np.savez_compressed(fd, labels=labels, embeddings=embeddings)

    def __call__(self):
        self.initialize()
        self.get_dataset()
        with tf.Graph().as_default():
            self.build_graph()
            self.build_saver()
            with create_session() as sess:
                self.run_last(sess)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # testing parameters
    argp.add_argument('dataset')
    argp.add_argument('--num-epochs', type=int, default=1)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--device', default='/gpu:0')
    argp.add_argument('--postfix', default='')
    argp.add_argument('--train-dir', default='./train{postfix}.tmp')
    argp.add_argument('--test-dir', default='./test{postfix}.tmp')
    argp.add_argument('--model-file')
    argp.add_argument('--log-file', default='test.log')
    argp.add_argument('--batch-size', type=int, default=72)
    # data parameters
    argp.add_argument('--dtype', type=int, default=2)
    argp.add_argument('--data-format', default='NCHW')
    argp.add_argument('--in-channels', type=int, default=1)
    argp.add_argument('--out-channels', type=int, default=5994)
    # pre-processing parameters
    Data.add_arguments(argp)
    argp.set_defaults(shuffle=False)
    # model parameters
    SRNTest.add_arguments(argp)
    # parse
    args = argp.parse_args(argv)
    args.train_dir = args.train_dir.format(postfix=args.postfix)
    args.test_dir = args.test_dir.format(postfix=args.postfix)
    args.dtype = [tf.int8, tf.float16, tf.float32, tf.float64][args.dtype]
    # run testing
    test = Test(args)
    test()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
