from pydub import AudioSegment
import numpy as np
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
        self.out_channels = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.get_files(self.out_channels)

    @staticmethod
    def add_arguments(argp):
        # pre-processing parameters
        argp.add_argument('--processes', type=int, default=4)
        argp.add_argument('--threads', type=int, default=5)
        argp.add_argument('--prefetch', type=int, default=64)
        argp.add_argument('--buffer-size', type=int, default=1024)

    def get_files(self, num_ids=None):
        dataset_ids = os.listdir(self.dataset)[:num_ids]
        dataset_ids = [os.path.join(self.dataset, i) for i in dataset_ids]
        data_list = []
        for i in range(num_ids):
            files = listdir_files(dataset_ids[i], filter_ext=['.wav', '.m4a'])
            for f in files:
                data_list.append((i, f))
        random.shuffle(data_list)
        # val set
        if self.val_size is not None:
            self.val_set = data_list[:self.val_size]
            data_list = data_list[self.val_size:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
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
    def process_sample(id_, file, num_labels):
        # parameters
        sample_rate = 16000
        slice_duration = 2000
        slice_samples = slice_duration * sample_rate // 1000
        # read from file
        audio = AudioSegment.from_file(file)
        #channels = audio.channels
        #sample_bytes = audio.frame_width
        #sample_rate = audio.frame_rate
        #duration = int(audio.duration_seconds * 1000)
        #samples = int(audio.frame_count())
        # to np.array
        data = np.array(audio.get_array_of_samples(), copy=False)
        samples = data.shape[-1]
        # slice
        start = random.randint(0, samples - slice_samples)
        data = data[start : start + slice_samples]
        # normalization
        norm_factor = 1 / audio.max
        data = data.astype(np.float32) * norm_factor
        # convert to CHW format
        data = np.expand_dims(np.expand_dims(data, 0), 0)
        # one-hot label
        label = np.zeros((num_labels,), np.float32)
        label[id_] = 1
        # return
        return data, label

    def extract_batch(self, batch_set):
        from concurrent.futures import ThreadPoolExecutor
        # initialize
        inputs = []
        labels = []
        # load all the data
        if self.threads == 1:
            for id_, file in batch_set:
                data, label = Data.process_sample(id_, file, self.out_channels)
                inputs.append(data)
                labels.append(label)
        else:
            with ThreadPoolExecutor(self.threads) as executor:
                futures = []
                for id_, file in batch_set:
                    futures.append(executor.submit(Data.process_sample, id_, file, self.out_channels))
                # final data
                for future in futures:
                    data, label = future.result()
                    inputs.append(data)
                    labels.append(label)
        # stack data to form a batch
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        return inputs, labels

    def gen_main(self, start=0):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // self.epoch_steps, self.num_epochs):
                step_offset = self.epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(self.epoch_steps, self.max_steps - step_offset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    batch_set = self.main_set[offset : offset + self.batch_size]
                    futures.append(executor.submit(self.extract_batch, batch_set))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        future = futures.pop(0)
                        yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def get_val(self):
        return self.extract_batch(self.val_set)
