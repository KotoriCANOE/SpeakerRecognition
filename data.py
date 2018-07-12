from pydub import AudioSegment
import numpy as np
from scipy import ndimage
from scipy.io import wavfile
import os
import random
from utils import eprint, listdir_files

class Data:
    def __init__(self, config):
        self.dataset = None
        self.num_epochs = None
        self.max_steps = None
        self.batch_size = None
        self.val_size = None
        self.processes = None
        self.threads = None
        self.prefetch = None
        self.buffer_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.get_files()

    @staticmethod
    def add_arguments(argp):
        # pre-processing parameters
        argp.add_argument('--processes', type=int, default=2)
        argp.add_argument('--threads', type=int, default=4)
        argp.add_argument('--prefetch', type=int, default=256)
        argp.add_argument('--buffer-size', type=int, default=1024)

    @staticmethod
    def group_shuffle(dataset, batch_size, shuffle=False, group_size=4):
        if shuffle:
            random.shuffle(dataset)
        upper = len(dataset)
        assert batch_size % group_size == 0
        batch_groups = batch_size // group_size
        for i in range(0, upper, batch_size):
            j = i + batch_groups
            ids = [d[0] for d in dataset[i : j]]
            counts = {ids[i]: group_size - 1 for i in range(batch_groups)}
            for k in range(j, upper):
                data = dataset[k]
                id_ = data[0]
                if id_ in ids and counts[id_] > 0:
                    counts[id_] -= 1
                    if counts[id_] <= 0:
                        ids.remove(id_)
                    dataset[k] = dataset[j]
                    dataset[j] = data
                    j += 1
                    if j >= i + batch_size:
                        break
        return dataset

    def get_files(self, num_ids=None):
        dataset_ids = os.listdir(self.dataset)[:num_ids]
        num_ids = len(dataset_ids)
        self.num_ids = num_ids
        dataset_ids = [os.path.join(self.dataset, i) for i in dataset_ids]
        data_list = []
        for i in range(num_ids):
            files = listdir_files(dataset_ids[i], filter_ext=['.wav', '.m4a'])
            for f in files:
                data_list.append((i, f))
        self.group_shuffle(data_list, self.batch_size, True)
        # val set
        if self.val_size is not None:
            assert self.val_size < len(data_list)
            self.val_steps = self.val_size // self.batch_size
            self.val_size = self.val_steps * self.batch_size
            self.val_set = data_list[:self.val_size]
            data_list = data_list[self.val_size:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
        assert self.batch_size <= len(data_list)
        self.epoch_steps = len(data_list) // self.batch_size
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = data_list[:self.epoch_size]
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(len(self.main_set), self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def process_sample(id_, file):
        # parameters
        slice_duration = 1000
        # read from file
        if os.path.splitext(file)[1] == '.wav':
            sample_rate, audio = wavfile.read(file)
            data = audio
            audio_max = np.max(data)
        else:
            audio = AudioSegment.from_file(file)
            #channels = audio.channels
            sample_rate = audio.frame_rate
            audio_max = audio.max
            # to np.array
            data = np.array(audio.get_array_of_samples(), copy=False)
        # slice
        samples = data.shape[-1]
        slice_samples = slice_duration * sample_rate // 1000
        if samples > slice_samples:
            start = random.randint(0, samples - slice_samples)
            data = data[start : start + slice_samples]
        # normalization
        norm_factor = 1 / audio_max
        data = data.astype(np.float32) * norm_factor
        # random data manipulation
        data = DataPP.process(data)
        # convert to CHW format
        data = np.expand_dims(np.expand_dims(data, 0), 0)
        # label
        label = np.array(id_, dtype=np.int64)
        # return
        return data, label

    @classmethod
    def extract_batch(cls, batch_set, threads=1):
        from concurrent.futures import ThreadPoolExecutor
        # initialize
        inputs = []
        labels = []
        # load all the data
        if threads == 1:
            for id_, file in batch_set:
                _input, _label = cls.process_sample(id_, file)
                inputs.append(_input)
                labels.append(_label)
        else:
            with ThreadPoolExecutor(threads) as executor:
                futures = []
                for id_, file in batch_set:
                    futures.append(executor.submit(cls.process_sample, id_, file))
                # final data
                for future in futures:
                    _input, _label = future.result()
                    inputs.append(_input)
                    labels.append(_label)
        # stack data to form a batch
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels

    def _gen_batches(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # grouping random shuffle
                if epoch > 0:
                    _dataset = dataset.copy()
                    self.group_shuffle(_dataset, self.batch_size, shuffle)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    upper = min(len(_dataset), offset + self.batch_size)
                    batch_set = _dataset[offset : upper]
                    futures.append(executor.submit(self.extract_batch, batch_set, self.threads))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        future = futures.pop(0)
                        yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def gen_main(self, start=0):
        return self._gen_batches(self.main_set, self.epoch_steps, self.num_epochs,
            start, True)

    def gen_val(self, start=0):
        return self._gen_batches(self.val_set, self.val_steps, 1,
            start, False)

class DataPP:
    @classmethod
    def process(cls, data):
        # smoothing
        smooth_prob = 0.1
        smooth_std = 0.75
        if cls.active_prob(smooth_prob):
            smooth_scale = cls.truncate_normal(smooth_std)
            data = ndimage.gaussian_filter1d(data, smooth_scale, truncate=2.0)
        # add noise
        noise_prob = 0.7
        noise_std = 0.025
        noise_smooth_prob = 0.8
        noise_smooth_std = 1.5
        while cls.active_prob(noise_prob):
            # Gaussian noise
            noise_scale = cls.truncate_normal(noise_std)
            noise = np.random.normal(0.0, noise_scale, data.shape)
            # noise smoothing
            if cls.active_prob(noise_smooth_prob):
                smooth_scale = cls.truncate_normal(noise_smooth_std)
                noise = ndimage.gaussian_filter1d(noise, smooth_scale, truncate=2.0)
            # add noise
            data += noise
        # random amplitude
        data *= 0.1 ** np.random.uniform(0, 2) # 0~-20 dB
        # return
        return data

    @staticmethod
    def active_prob(prob):
        return np.random.uniform(0, 1) < prob

    @staticmethod
    def truncate_normal(std, mean=0.0, max_rate=4.0):
        max_scale = std * max_rate
        scale = max_scale + 1.0
        while scale > max_scale:
            scale = np.abs(np.random.normal(0.0, std))
        scale += mean
        return scale
