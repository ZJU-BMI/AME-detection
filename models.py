import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import time


class BasicLSTMModel(object):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='BasicLSTMModel'):
        """

        :param num_features: dimension of input data per time step
        :param time_steps: max time step
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
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._lasso = lasso
        self._ridge = ridge

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace)

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
            # TODO 后续加入正则
            if lasso != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l1_regularizer(lasso)(trainable_variables)
            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)

    def _hidden_layer(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)  # ??????
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)  # 全零向量

        mask, length = self._length()  # 每个病人的实际天数
        self._hidden, _ = tf.nn.dynamic_rnn(lstm,
                                            self._x,
                                            sequence_length=length,
                                            initial_state=init_state)
        self._hidden_rep = tf.reduce_sum(self._hidden, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True),
                                                                    (1, self._lstm_size))

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._x), 2))  # 每个step若有实际数据则为1，只有填零数据则为0
        length = tf.reduce_sum(mask, 1)  # 每个sample的实际step数
        length = tf.cast(length, tf.int32)  # 类型转换
        return mask, length

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("auc\tepoch\tloss\tloss_diff\tcount")
        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                auc = roc_auc_score(test_set.labels, y_score)
                print("{}\t{}\t{}\t{}\t{}".format(auc, data_set.epoch_completed, loss, loss_diff, count),
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # 训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_feature})


class BidirectionalLSTMModel(BasicLSTMModel):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='Bi-LSTM'):
        super().__init__(num_features, time_steps, lstm_size, n_output, batch_size, epochs, output_n_epoch,
                         learning_rate, max_loss, max_pace, lasso, ridge, optimizer, name)  # 调用父类BasicLSTMModel的初始化函数

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


class ContextAttentionRNN(BidirectionalLSTMModel):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='BasicLSTMModel'):

        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch
        self._lstm_size = lstm_size
        self._time_steps = time_steps
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._lasso = lasso
        self._ridge = ridge

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')
            self._v = tf.placeholder(tf.float32, [None, time_steps, num_features, 10], "contextual_words")

            self._sess = tf.Session()  # 会话

            self._W_trans = tf.Variable(tf.truncated_normal([1, 100], stddev=0.1))

            self._attention_mechanism()

            self._hidden_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.sigmoid(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._output),
                                        name='loss')
            # TODO 后续加入正则
            if lasso != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l1_regularizer(lasso)(trainable_variables)
            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)

        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._context,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden_concat = tf.concat(self._hidden, axis=2)  # n_samples×time_steps×2lstm_size→n_samples×2lstm_size
        self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True),
                                                                           (1, self._lstm_size * 2))

    # TODO 后续再做调整：1.time_step在原有基础上+10（attention机制后前后各多出5个）；2.attention计算时考虑长度，减计算量
    def _attention(self, x):
        word_embedding_matrix = np.zeros([x.shape[0], self._time_steps + 10, 100], dtype=np.float32)
        word_embedding_matrix[:, 5:self._time_steps + 5, :] = x.reshape([-1, self._time_steps, 100])
        context_embedding_matrix = np.zeros([x.shape[0], self._time_steps, 100])
        length = np.sum(np.sign(np.max(np.abs(x), 2)), 1).astype(np.int32)
        for j in range(x.shape[0]):
            for k in range(np.min((length[j] + 5, 80))):  # 此处暂时按照原版验证效果是否一致，之后再做修改
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

    def _attention_mechanism(self):
        # xp = tf.concat(
        #     [tf.concat([tf.zeros([5, self._num_features]), self._x], 0), tf.zeros([None,5, self._num_features])], 0)
        # zeros = tf.constant(tf.zeros([None, 5, self._num_features], dtype=tf.float32))
        # xp = tf.concat([zeros, self._x], 1)
        W_x = tf.tile(self._W_trans, [self._time_steps, 1])
        W_v = tf.tile(tf.reshape(self._W_trans, [1, -1, 1]), [self._time_steps, 1, 10])
        x_trans = tf.nn.tanh(tf.multiply(self._x, tf.tile(self._W_trans, [self._time_steps, 1])))
        v_trans = tf.nn.tanh(tf.multiply(self._v, W_v))
        a = tf.matmul(tf.reshape(x_trans, [-1, self._time_steps, 1, self._num_features]), v_trans)
        z = tf.transpose(tf.nn.softmax(a), [0, 1, 3, 2])
        context = tf.matmul(self._v, z)
        self._context = tf.reshape(context,[-1,self._time_steps,self._num_features])

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("auc\tepoch\tloss\tloss_diff\tcount")
        # data_set = DataSet(self._attention(data_set), data_set.labels)

        logged = set()
        loss = 0
        count = 0
        # TODO 迭代停止条件改写已完成， 若试参可逐步显示各指标
        while data_set.epoch_completed < self._epochs:
            dynamic_feature, labels = data_set.next_batch(self._batch_size)
            # dynamic_feature = self._attention(dynamic_feature)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                auc = roc_auc_score(test_set.labels, y_score)
                print("{}\t{}\t{}\t{}\t{}".format(auc, data_set.epoch_completed, loss, loss_diff, count),
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # 训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break


class LogisticRegression(object):
    # TODO 所有模型learning_rate需改，在experiment中--已完成
    def __init__(self, num_features, time_steps, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=2.0, max_pace=0.1, lasso=0.0, ridge=0.0, optimizer=tf.train.AdamOptimizer,
                 name='LRModel'):
        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch
        self._time_steps = time_steps
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._lasso = lasso
        self._ridge = ridge
        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace)
        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps * num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            self._sess = tf.Session()  # 会话

            self._hidden_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.sigmoid(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._output),
                                        name='loss')
            if lasso != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l1_regularizer(lasso)(trainable_variables)
            if ridge != 0:
                for trainable_variables in tf.trainable_variables(self._name):
                    self._loss += tf.contrib.layers.l2_regularizer(ridge)(trainable_variables)

            self._train_op = optimizer(learning_rate).minimize(self._loss)

    def _hidden_layer(self):
        self._hidden_rep = self._x

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        for c in tf.trainable_variables(self._name):
            print(c.name)

        print("auc\tepoch\tloss\tloss_diff\tcount")

        logged = set()
        loss = 0
        count = 0
        while data_set.epoch_completed < self._epochs:
            dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op,
                           feed_dict={self._x: dynamic_feature.reshape([-1, self._time_steps * self._num_features]),
                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={
                    self._x: data_set.dynamic_feature.reshape([-1, self._time_steps * self._num_features]),
                    self._y: data_set.labels})
                loss_diff = loss_prev - loss
                y_score = self.predict(test_set)
                auc = roc_auc_score(test_set.labels, y_score)
                print("{}\t{}\t{}\t{}\t{}".format(auc, data_set.epoch_completed, loss, loss_diff, count),
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # 训练停止条件
                if loss > self._max_loss:
                    count = 0
                else:
                    if loss_diff > self._max_pace:
                        count = 0
                    else:
                        count += 1
                if count > 9:
                    break

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={
            self._x: test_set.dynamic_feature.reshape([-1, self._time_steps * self._num_features])})


class MultiLayerPerceptron(LogisticRegression):
    def __init__(self, num_features, time_steps, hidden_units, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=2.0, max_pace=0.1, lasso=0.0, ridge=0.0, optimizer=tf.train.AdamOptimizer,
                 name='MLPModel'):
        self._hidden_units = hidden_units
        super().__init__(num_features, time_steps, n_output, batch_size, epochs, output_n_epoch, learning_rate,
                         max_loss, max_pace, lasso, ridge, optimizer, name)

    def _hidden_layer(self):
        self._hidden_rep = tf.nn.relu(tf.contrib.layers.fully_connected(self._x, self._hidden_units))
