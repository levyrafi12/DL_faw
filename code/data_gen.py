import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.utils import Sequence

infected_threshold = 0.01

def infest_to_class_ind(n_classes, infest):
    if n_classes > 2:
        if infest > 0:
            return min(int(infest / (1 / (n_classes - 1))) + 1, n_classes - 1)
        return 0
    return 1 if infest > infected_threshold else 0

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, root_dir, image_infest_list, batch_size, dim, n_classes):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.image_infest_list = image_infest_list
        self.epoch = 0 # used for the seed
        self.root_dir = root_dir
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        n_samples = len(self.image_infest_list)
        n_batches = int(n_samples / self.batch_size)
        if n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        x_vecs = []
        y_labels = []
        low = index * self.batch_size
        high = min((index + 1) * self.batch_size, len(self.image_infest_list))
        for i in range(low, high):
            img_fn = self.image_infest_list[i][0]
            img = cv2.imread(self.root_dir + img_fn)
            assert not img is None, img_fn
            assert img.any() != None, img_fn
            # print(root_dir + img_fn)
            img = np.array(img) / 255.0 
            if img.shape[0] < img.shape[1]: # height first
                img = np.transpose(img, (1,0,2))
            img = cv2.resize(img, (self.dim[0], self.dim[1]))
            # plt.imshow(img)
            # plt.show()
            label = infest_to_class_ind(self.n_classes, self.image_infest_list[i][1])
            y_labels.append(label)
            x_vecs.append(img.reshape((1,) + img.shape))

        y_labels = to_categorical(y_labels, self.n_classes)
        x_vecs = np.vstack(x_vecs)
        return x_vecs, y_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.epoch += 1
        np.random.seed(self.epoch)
        np.random.shuffle(self.image_infest_list)