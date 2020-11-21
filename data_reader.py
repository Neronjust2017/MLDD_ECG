import h5py
import numpy as np

from utils import unfold_label, shuffle_data


class BatchEcgGenerator:
    def __init__(self, flags, stage, dataset, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage)
        self.get_data(dataset, b_unfold_label)

    def configuration(self, flags, stage):
        self.batch_size = flags.batch_size
        self.current_index = -1
        self.stage = stage
        self.shuffled = False


    def get_data(self, dataset, b_unfold_label):
        self.ecgs = dataset[0]
        self.labels = dataset[1]

        # shift the labels to start from 0 : already done.

        if b_unfold_label:
            self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage is 'train':
            self.ecgs, self.labels = shuffle_data(samples=self.ecgs, labels=self.labels)


    def get_ecgs_labels_batch(self):

        ecgs = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.ecgs, self.labels = shuffle_data(samples=self.ecgs, labels=self.labels)

            ecgs.append(self.ecgs[self.current_index])
            labels.append(self.labels[self.current_index])

        ecgs = np.stack(ecgs)
        labels = np.stack(labels)

        return ecgs, labels