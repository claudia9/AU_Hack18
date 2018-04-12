import pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from os import makedirs
from os.path import exists

# The data directory that the raw data is stored in
data_dir = './data_set/MNIST_data'
# The npy file with 'normalized' data
data_file = 'mnist.pkl'

class DataFeed(object):
    def __init__(self):
        self.data = fetch_data()
        self.max = self.data['train_images'].shape[0]
        self.val_max = self.data['validation_images'].shape[0]
        self.test_max = self.data['test_images'].shape[0]
        self.shuffle = True
        self.permutation = []
        self.offset = 0
        self.reset_permutation()

    def reset_permutation(self):
        if self.shuffle:
            self.permutation = np.random.permutation(self.max)

        self.offset = 0

    def next(self, batch_size):
        start = self.offset
        self.offset = min(self.offset + batch_size, self.max)

        selection = self.permutation[start:self.offset]

        if self.offset == self.max:
            print("resetting")
            self.reset_permutation()

        return self.data['train_images'][selection], self.data['train_labels'][selection]

    def train(self, size=5000):
        selection = np.random.permutation(self.max)[:size]
        return self.data['train_images'][selection], self.data['train_labels'][selection]

    def validation(self, size=5000):
        selection = np.random.permutation(self.val_max)[:size]
        return self.data['validation_images'][selection], self.data['validation_labels'][selection]

    def test(self):
        return self.data['test_images'], self.data['test_labels']

# Used to ignore division by 0 when normalizing data
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

# Either load it from disk or download it
def fetch_data():
    if not exists(data_dir):
        makedirs(data_dir)

    # Normalize data once if we haven't done it before and store it in a file
    if not exists(f'{data_dir}/{data_file}'):
        print('Downloading MNIST data')
        mnist = input_data.read_data_sets(data_dir, one_hot=True)

        def _normalize(data, mean=None, std=None):
            if mean is None:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
            return div0((data - mean), std), mean, std

        train_data, mean, std = _normalize(mnist.train.images)

        validation_data, *_ = _normalize(mnist.validation.images, mean, std)
        test_data, *_ = _normalize(mnist.test.images, mean, std)

        mnist_data = {'train_images': train_data,
                      'train_labels': mnist.train.labels,
                      'validation_images': validation_data,
                      'validation_labels': mnist.validation.labels,
                      'test_images': test_data,
                      'test_labels': mnist.test.labels}
        with open(f'{data_dir}/{data_file}', 'wb') as f:
            pickle.dump(mnist_data, f)

    # If we have normalized the data already; load it
    else:
        with open(f'{data_dir}/{data_file}', 'rb') as f:
            mnist_data = pickle.load(f)

    return mnist_data


if __name__ == '__main__':
    print(fetch_data())
