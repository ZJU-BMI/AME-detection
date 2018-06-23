import numpy as np
import pickle


class DataSet(object):
    def __init__(self, dynamic_feature, labels):
        self._dynamic_feature = dynamic_feature
        self._labels = labels
        self._num_examples = labels.shape[0]
        self._epoch_completed = 0
        self._batch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self.num_examples or batch_size <= 0:
            # raise ValueError('The size of one batch: {} should be less than the total number of '
            #                  'data: {}'.format(batch_size, self.num_examples))
            batch_size = self._labels.shape[0]
        if self._batch_completed == 0:
            self._shuffle()
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self.num_examples:
            self._epoch_completed += 1
            # TODO 没必要拼接，直接输入剩余部分即可（或者舍弃亦可）
            dynamic_rest_part = self._dynamic_feature[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]

            self._shuffle()  # 打乱,在一个新的epoch里重新打乱
            self._index_in_epoch = 0
            return dynamic_rest_part, label_rest_part
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

    @property
    def batch_completed(self):
        return self._batch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


# def read_data(event_type):
#     # TODO 字符串改为constant
#     dynamic_features = pickle.load(open("resources/input_file_progress_notes.pkl", "rb"))
#     if event_type == "qx":
#         labels = pickle.load(open("resources/output_file_progress_notes_qx.pkl", "rb"))[:, -1].reshape([-1, 1])
#     elif event_type == "xycj":
#         labels = pickle.load(open("resources/output_file_progress_notes_xycj.pkl", "rb"))[:, -1].reshape([-1, 1])
#     else:
#         labels = pickle.load(open("resources/output_file_progress_notes_cx.pkl", "rb"))[:, -1].reshape([-1, 1])
#     return DataSet(dynamic_features, labels)


def read_data(event_type):
    # TODO 字符串改为constant
    dynamic_features = pickle.load(open("resources/input_file_admission_records.pkl", "rb"))
    if event_type == "qx":
        labels = pickle.load(open("resources/output_file_admission_records_qx.pkl", "rb"))[:, -1].reshape([-1, 1])
    elif event_type == "xycj":
        labels = pickle.load(open("resources/output_file_admission_records_xycj.pkl", "rb"))[:, -1].reshape([-1, 1])
    else:
        labels = pickle.load(open("resources/output_file_admission_records_cx.pkl", "rb"))[:, -1].reshape([-1, 1])
    return DataSet(dynamic_features, labels)


if __name__ == "__main__":
    read_data()
