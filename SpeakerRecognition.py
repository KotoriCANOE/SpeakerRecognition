import tensorflow as tf
import numpy as np
import pandas as pd
import os
import librosa

# setup tensorflow and return session
def create_session():
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    return tf.Session(config=config)

class AudioLoader:
    def __init__(self, rate=16000, length=8.0, batch_size=4, processes=4, prefetch=16):
        self.rate = rate
        self.length = length
        self.batch_size = batch_size
        self.processes = processes
        self.prefetch = prefetch

    @staticmethod
    def load_file(file, rate, length):
        num_samples = int(rate * length + 0.5)
        audio, _rate = librosa.load(file, sr=rate, mono=True, duration=length)
        if audio.shape[0] < num_samples:
            audio = np.pad(audio, (0, num_samples - audio.shape[0]), 'constant')
        return audio

    @classmethod
    def load_batch(cls, files, rate, length):
        audio_batch = [cls.load_file(file, rate, length) for file in files]
        return audio_batch

    def __call__(self, inputs):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            for offset in range(0, len(inputs), self.batch_size):
                upper = min(len(inputs), offset + self.batch_size)
                files = inputs[offset : upper]
                futures.append(executor.submit(self.load_batch, files, self.rate, self.length))
                # yield the data beyond prefetch range
                while len(futures) >= self.prefetch:
                    future = futures.pop(0)
                    yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

class TimbreNet:
    def __init__(self, model_dir, device='/gpu:0'):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = create_session()
            with tf.device(device):
                self.load_model(model_dir)

    def __del__(self):
        self.sess.close()

    def load_model(self, model_dir):
        saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta'), clear_devices=True)
        if saver is None:
            raise ValueError('Failed to import meta graph!')
        saver.restore(self.sess, os.path.join(model_dir, 'model'))
        # access input/output variables
        self.inputs = self.graph.get_tensor_by_name('Input:0')
        self.embeddings = self.graph.get_tensor_by_name('Embedding:0')
        # finalize the graph
        self.graph.finalize()

    # input NCHW format array
    def inference(self, inputs):
        fetch = self.embeddings
        feed_dict = {self.inputs: inputs}
        ret = self.sess.run(fetch, feed_dict)
        return ret

    # input a batch of CHW format array
    def process_batch(self, inputs):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        while len(inputs.shape) < 4:
            inputs = np.expand_dims(inputs, -2)
        embeddings = self.inference(inputs)
        return embeddings
    
    # split a batch array into a list of arrays
    @staticmethod
    def split_batch(batch):
        return [b for b in batch]

    # input a list of audio files
    def __call__(self, ifiles, loader=None):
        import time
        # audio loader and inputs generator
        if loader is None:
            loader = AudioLoader()
        inputs_gen = loader(ifiles)
        # iterate over batches
        tick = time.time()
        embeddings = []
        for inputs in inputs_gen:
            _embeddings = self.process_batch(inputs)
            embeddings.append(_embeddings)
        tock = time.time()
        # concat over batches
        embeddings = np.concatenate(embeddings, axis=0)
        print('Processed {} samples in {}s'.format(len(embeddings), tock - tick))
        return embeddings

class Clusterer:
    def __init__(self):
        self.clusterer = None

    @staticmethod
    def metrics(pred, labels=None, embeddings=None):
        from sklearn import metrics
        print('Estimated number of clusters: {}'.format(len(np.unique(pred))))
        if labels is not None:
            print("Homogeneity: {:0.3f}".format(metrics.homogeneity_score(labels, pred)))
            print("Completeness: {:0.3f}".format(metrics.completeness_score(labels, pred)))
            print("V-measure: {:0.3f}".format(metrics.v_measure_score(labels, pred)))
            print("Adjusted Rand Index: {:0.3f}"
                .format(metrics.adjusted_rand_score(labels, pred)))
            print("Adjusted Mutual Information: {:0.3f}"
                .format(metrics.adjusted_mutual_info_score(labels, pred)))
        if embeddings is not None:
            print("Silhouette Coefficient: {:0.3f}"
                .format(metrics.silhouette_score(embeddings, pred)))
            print("Calinski-Harabaz Index: {:0.3f}"
                .format(metrics.calinski_harabaz_score(embeddings, pred)))

    @classmethod
    def optimize_clusters(cls, X, start, stop, silhouettes=None, random_state=0):
        import math
        from sklearn import cluster, metrics
        if silhouettes is None:
            silhouettes = [0] * (stop + 1)
        # loop over multiple number of clusters to calculate the score
        step = max(1, int(math.sqrt(stop - start) + 0.5))
        for n_clusters in range(start, stop + 1, step):
            if silhouettes[n_clusters] > 0:
                continue
            clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = clusterer.fit_predict(X)
            silhouette = metrics.silhouette_score(X, cluster_labels)
            silhouettes[n_clusters] = silhouette
            print('n_clusters={}, silhouette={}'.format(n_clusters, silhouette))
        # return if step is 1
        if step <= 1:
            argmax = np.argmax(silhouettes)
            return argmax
        # get the top-2 arg as range for the next iteration
        top2 = np.argsort(silhouettes[start : stop + 1])[-2:] + start
        start2, stop2 = min(top2) + 1, max(top2)
        return cls.optimize_clusters(X, start2, stop2, silhouettes, random_state)

    def __call__(self, embeddings, min_cls, max_cls):
        from sklearn import cluster
        n_clusters = min_cls if min_cls == max_cls else self.optimize_clusters(embeddings, min_cls, max_cls)
        self.clusterer = cluster.KMeans(n_clusters, random_state=0).fit(embeddings)
        cluster_labels = self.clusterer.labels_
        self.metrics(cluster_labels, None, embeddings)
        return cluster_labels

    def predict(self, embeddings):
        if self.clusterer is None:
            raise ValueError('Should first fit the clusterer before predicting.')
        return self.clusterer.predict(embeddings)

def process_cluster(args, ifiles, opath, relative=True):
    if not os.path.exists(opath):
        os.makedirs(opath)
    embeddings = args.timbrenet(ifiles)
    # relative/absolute path
    if relative:
        ifiles = [os.path.relpath(f, opath) for f in ifiles]
    else:
        ifiles = [os.path.abspath(f) for f in ifiles]
    # save embeddings
    with open(os.path.join(opath, 'embeddings.npz'), 'wb') as fd:
        np.savez_compressed(fd, files=ifiles, embeddings=embeddings)
    # save clustering results
    if args.cluster:
        clusterer = Clusterer()
        labels = clusterer(embeddings, args.cluster_min, args.cluster_max)
        with open(os.path.join(opath, 'cluster.csv'), 'w') as fd:
            df = {'file': ifiles, 'label': labels}
            df = pd.DataFrame(df)
            df.to_csv(fd)

def process(args):
    # find all the audio files in the given path
    postfix = ['.wav', '.m4a', '.mp3']
    file_dict = {}
    for dirpath, dirnames, filenames in os.walk(args.input):
        filenames = [f for f in filenames if os.path.splitext(f)[1] in postfix]
        if len(filenames) > 0:
            file_dict[dirpath] = filenames
        if not args.recursive:
            break
    # initialize
    args.timbrenet = TimbreNet(args.model_dir, args.device)
    # process each dir
    if args.output is None: # save results to each directory
        for dirpath in file_dict.keys():
            ifiles = [os.path.join(dirpath, f) for f in file_dict[dirpath]]
            process_cluster(args, ifiles, dirpath, args.relative)
    else: # save all the results to the specific directory
        ifiles = []
        for dirpath in file_dict.keys():
            ifiles += [os.path.join(dirpath, f) for f in file_dict[dirpath]]
        process_cluster(args, ifiles, args.output, args.relative)

def add_arguments(argp):
    def bool_argument(argp, name, default, help=None):
        argp.add_argument('--' + name, dest=name, action='store_true', help=help)
        argp.add_argument('--no-' + name, dest=name, action='store_false', help=help)
        eval('argp.set_defaults({}={})'.format(name, 'True' if default else 'False'))
    argp.add_argument('input')
    bool_argument(argp, 'recursive', False,
        help='recursively process sub-directories')
    argp.add_argument('-o', '--output',
        help='optional output directory instead of saving results to each sub-directory')
    bool_argument(argp, 'relative', True,
        help='save relative/absolute path')
    argp.add_argument('--model-dir', default='model',
        help='directory to load the model files')
    argp.add_argument('--device', default='/gpu:0')
    bool_argument(argp, 'cluster', True,
        help='whether to cluster the embeddings')
    argp.add_argument('--cluster-min', type=int, default=8,
        help='min number of clusters for clustering')
    argp.add_argument('--cluster-max', type=int, default=40,
        help='max number of clusters for clustering')

def main(argv):
    import argparse
    argp = argparse.ArgumentParser()
    add_arguments(argp)
    args = argp.parse_args(argv)
    process(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
