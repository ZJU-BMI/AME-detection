import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Imputer, MinMaxScaler


class DataSet(object):
    def __init__(self, dynamic_feature, labels):
        self._dynamic_feature = dynamic_feature
        self._labels = labels
        self._num_examples = labels.shape[0]
        self._epoch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self.num_examples or batch_size <= 0:
            # raise ValueError('The size of one batch: {} should be less than the total number of '
            #                  'data: {}'.format(batch_size, self.num_examples))
            batch_size = self._labels.shape[0]
        start = self._index_in_epoch
        if start + batch_size > self.num_examples:
            self._epoch_completed += 1
            rest_num_examples = self._num_examples - start  # 旧epoch中剩余部分
            dynamic_rest_part = self._dynamic_feature[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]

            self._shuffle()  # 打乱,在一个新的epoch里重新打乱
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            dynamic_new_part = self._dynamic_feature[start:end]
            label_new_part = self._labels[start:end]
            return (np.concatenate((dynamic_rest_part, dynamic_new_part), axis=0),
                    np.concatenate((label_rest_part, label_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_feature[start:end], self._labels[start:end]

    def _shuffle(self):  # 打乱
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._dynamic_feature = self.dynamic_feature[index]
        self._labels = self.labels[index]

    @property
    def dynamic_feature(self):
        return self._dynamic_feature

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


class DataSetWithContext(DataSet):
    def __init__(self, dynamic_feature, labels):
        super().__init__(dynamic_feature, labels)
        self._contextual_words()
        self._context = self._contextual_words()

    def _contextual_words(self):
        shape = self.dynamic_feature.shape
        contextual_word_embedding_matrix = np.zeros([shape[0], shape[1], shape[2], 10])
        padded_word_embedding_maxtrix = np.zeros([shape[0], shape[1] + 10, shape[2]])
        padded_word_embedding_maxtrix[:, 5:shape[1] + 5, :] = self.dynamic_feature
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(5):
                    contextual_word_embedding_matrix[i, j, :, k] = np.transpose(
                        padded_word_embedding_maxtrix[i, j + k, :])
                    contextual_word_embedding_matrix[i, j, :, k + 5] = np.transpose(
                        padded_word_embedding_maxtrix[i, j + k + 6, :])
        return contextual_word_embedding_matrix

    @property
    def context(self):
        return self._context


def read_data():
    # TODO 改为switch形式，选择读取不同事件
    dynamic_features = pickle.load(open("input_file_np_1000f_xjqx+_80w.pkl", "rb"))
    labels = pickle.load(open("output_file_np_1000f_xjqx+_80w.pkl", "rb"))[:, -1].reshape([-1, 1])
    return DataSet(dynamic_features, labels)


if __name__ == "__main__":
    read_data()
