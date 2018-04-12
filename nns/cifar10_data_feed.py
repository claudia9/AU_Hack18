import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys

class DataFeed(object):
    def __init__(self):
        train_images, train_labels, train_descriptions = get_data_set()
        val_images, val_labels, val_descriptions = get_data_set('test')
        self.data = {
            'train_images': train_images,
            'train_labels': train_labels,
            'train_descriptions': train_descriptions,
            'validation_images': val_images,
            'validation_labels': val_labels,
            'validation_descriptions': val_descriptions
        }

        self.max = self.data['train_images'].shape[0]
        self.val_max = self.data['validation_images'].shape[0]

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

    def train(self, size=100):
        selection = np.random.permutation(self.max)[:size]
        return self.data['train_images'][selection], self.data['train_labels'][selection]

    def validation(self, size=5000):
        selection = np.random.permutation(self.val_max)[:size]
        return self.data['validation_images'][selection], self.data['validation_labels'][selection]

    def test(self):
        return self.validation()


def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None

    maybe_download_and_extract()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), l


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(f'{main_directory}cifar_10'):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)
