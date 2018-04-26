import tensorflow as tf
import numpy as np
from data import DataSet


class BasicLSTMModel(object):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000,
                 output_n_epoch=10, optimizer=tf.train.AdamOptimizer(), name='BasicLSTMModel'):
        """

        :param num_features: dimension of input data per time step    time step的作用是什么？
        :param time_steps: max time step ？
        :param batch_size: batch size
        :param lstm_size: size of lstm cell
        :param n_output: classes
        :param epochs: epochs to train
        :param output_n_epoch: output loss per n epoch
        :param optimizer: optimizer
        :param name: model name
        """
        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch
        self._lstm_size = lstm_size
        self._time_steps = time_steps

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            self._sess = tf.Session()  # 会话

            self._hidden_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.sigmoid(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._output),
                                        name='loss')
            self._train_op = optimizer.minimize(self._loss)

    def _hidden_layer(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)  # ??????
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)  # 全零向量

        mask, length = self._length()  # 每个病人的实际天数
        self._hidden, _ = tf.nn.dynamic_rnn(lstm,
                                            self._x,
                                            sequence_length=length,
                                            initial_state=init_state)
        self._hidden_rep = tf.reduce_sum(self._hidden, 1) / tf.tile(tf.reduce_sum(mask, 1, keepdims=True),
                                                                    (1, self._lstm_size))

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._x), 2))  # 每个step若有实际数据则为1，只有填零数据则为0
        length = tf.reduce_sum(mask, 1)  # 每个sample的实际step数
        length = tf.cast(length, tf.int32)  # 类型转换
        return mask, length

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
        while data_set.epoch_completed < self._epochs:
            dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_feature})


class BidirectionalLSTMModel(BasicLSTMModel):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 optimizer=tf.train.AdamOptimizer(), name='bidirectionalLSTMModel'):
        super().__init__(num_features, time_steps, lstm_size, n_output, batch_size, epochs, output_n_epoch, optimizer,
                         name)  # 调用父类BasicLSTMModel的初始化函数

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)

        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._x,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden_concat = tf.concat(self._hidden, axis=2)  # n_samples×time_steps×2lstm_size→n_samples×2lstm_size
        self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True),
                                                                           (1, self._lstm_size * 2))


class CA_RNN(BasicLSTMModel):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 optimizer=tf.train.AdamOptimizer(), name="BiLSTMWithAttention"):
        super().__init__(num_features, time_steps, lstm_size, n_output, batch_size, epochs, output_n_epoch, optimizer,
                         name)
        with tf.variable_scope(self._name):
            self._W_trans = tf.Variable(tf.truncated_normal([10, 100], stddev=0.1))

    def _attention(self, dataset):
        x = dataset.dynamic_feature
        word_embedding_matrix = np.zeros([x.shape[0], self._time_steps + 10, 100], dtype=np.float32)
        word_embedding_matrix[:, 5:self._time_steps + 5, :] = x.reshape([-1, self._time_steps, 100])
        context_embedding_matrix = np.zeros([x.shape[0], self._time_steps, 100])
        for j in range(x.shape[0]):
            for k in range(self._time_steps):
                words_2n = np.zeros([10, 100], dtype=np.float32)
                word_k = word_embedding_matrix[j, k + 5, :].astype(np.float32)
                words_2n[0:5, :] = word_embedding_matrix[j, k:k + 5, :]
                words_2n[5:10, :] = word_embedding_matrix[j, k + 6:k + 11, :]
                w_c = self._W_trans.eval(self._sess)
                words_2n_trans = np.tanh(np.multiply(words_2n, w_c))
                z = np.zeros(10)
                for n in range(10):
                    z[n] = words_2n_trans[n:n + 1, :] @ word_k.reshape([len(word_k), 1])
                z = z - np.max(z)
                z_sm = np.exp(z) / np.sum(np.exp(z))
                context_embedding = np.matmul(z_sm, words_2n)
                context_embedding_matrix[j, k, :] = context_embedding
        return context_embedding_matrix

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0
        data_set = DataSet(self._attention(data_set), data_set.labels)

        logged = set()
        while data_set.epoch_completed < self._epochs:
            dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))
