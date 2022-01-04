import os
import math
import tifffile as tif
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, data_dir, shuffle=True, phase='train', test_size=0.1):
        # Data Generator configuration
        self.batch_size, self.shuffle = batch_size, shuffle
        self.image_files = os.listdir(os.path.join(data_dir, "images"))
        self.mask_files = os.listdir(os.path.join(data_dir, "masks"))

        if phase == 'train':
            # Select only 90% of the images for training
            self.image_files = self.image_files[:int(len(self.image_files) * (1.0 - test_size))]
            self.mask_files = self.mask_files[:int(len(self.mask_files) * (1.0 - test_size))]
        else:
            # Select only the last 10% of the images for validation
            self.image_files = self.image_files[int(len(self.image_files) * (1.0 - test_size)):]
            self.mask_files = self.mask_files[int(len(self.mask_files) * (1.0 - test_size)):]

    def __len__(self):
        return math.ceil(len(self.image_files) / self.batch_size)

    def __getitem__(self, index):
        files_x = self.image_files[index * self.batch_size: (index + 1) * self.batch_size]
        files_y = self.mask_files[index * self.batch_size: (index + 1) * self.batch_size]

        assert len(files_x) == len(files_y)

        batch_x, batch_y = [], []
        for i in range(len(files_x)):
            batch_x.append(tif.imread(os.path.join("data", "images", files_x[i])))
            batch_y.append(tif.imread(os.path.join("data", "masks", files_y[i])))

        batch_x = np.expand_dims(tf.keras.utils.normalize(np.array(batch_x), axis=1), 3)
        batch_y = np.expand_dims((np.array(batch_y)), 3) / 255.

        return batch_x, batch_y

    def on_epoch_end(self):
        # implement shuffling at the end of each epoch
        if self.shuffle:
            assert len(self.image_files) == len(self.mask_files)
            perm = np.random.permutation(len(self.image_files))
            self.image_files = self.image_files[perm]
            self.mask_files = self.mask_files[perm]