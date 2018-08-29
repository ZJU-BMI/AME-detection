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

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)

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

    def fit(self, data_set, test_set, event_type):
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

                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
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

    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


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
                 optimizer=tf.train.AdamOptimizer, name='CA-RNN'):

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
        self._template_format()

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')
            self._v = tf.placeholder(tf.int32, [time_steps, 10], "context_template")

            self._sess = tf.Session()  # 会话

            self._W_trans = tf.Variable(tf.truncated_normal([1, self._num_features], stddev=0.1))

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
            self._saver = tf.train.Saver()

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

    # TODO 后续再做调整：1.time_step在原有基础上+10（attention机制后前后各多出5个）

    def _template_format(self):
        template_i = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=np.int32).reshape([1, -1])
        add_one = np.ones([10], dtype=np.int32).reshape([1, -1])
        self._template = np.zeros([0, 10], dtype=np.int32)
        for i in range(self._time_steps):
            self._template = np.concatenate([self._template, template_i], 0)
            template_i = template_i + add_one

    def _attention_mechanism(self):
        W_x = tf.tile(self._W_trans, [self._time_steps, 1])
        W_v = tf.tile(tf.reshape(self._W_trans, [1, -1, 1]), [self._time_steps, 1, 10])
        # weights for nonlinear mapping

        v = tf.gather(tf.pad(self._x, [[0, 0], [5, 5], [0, 0]]), self._template, axis=1)
        # get the contextual words (word embeddings)

        x_trans = tf.nn.tanh(tf.multiply(self._x, W_x))
        c_trans = tf.nn.tanh(tf.multiply(tf.transpose(v, [0, 1, 3, 2]), W_v))
        # nonlinear mapping of word and their contextual words (word embeddings). Corresponding to equation (1)and (2)

        a = tf.matmul(tf.reshape(x_trans, [-1, self._time_steps, 1, self._num_features]), c_trans)
        self._z = tf.nn.softmax(a, 3)
        # the assigned attention weights of words, denoted as Z in our paper. Corresponding to equation (3)and (4)

        u = tf.reshape(tf.matmul(self._z, v), [-1, self._time_steps, self._num_features])
        # the weighted average of word embeddings, denoted as u i our paper. Corresponding to equation (5)

        self._context = tf.contrib.layers.fully_connected(tf.concat([u, self._x], 2), self._num_features)
        # the context embeddings, denoted as c in our paper. Corresponding to equation (6)

    def fit(self, data_set, test_set, event_type):
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
                                                      self._y: labels,
                                                      self._v: self._template})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                # 由于内存不足，在case2时只能计算一个mini-batch的loss
                loss = self._sess.run(self._loss, feed_dict={self._x: dynamic_feature,
                                                             self._y: labels,
                                                             self._v: self._template})
                loss_diff = loss_prev - loss

                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
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

        save_path = self._saver.save(self._sess,
                                     "model/" + event_type + "_case1_save_net" + time.strftime("%m-%d-%H-%M",
                                                                                               time.localtime()) + ".ckpt")
        print("Save to path: ", save_path)

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_feature, self._v: self._template})

    def attention_analysis(self, test_dynamic, model):
        # todo 输入为test_data，读取模型并返回attention权重
        saver = tf.train.Saver()
        saver.restore(self._sess, "model/" + model)
        prob = self._sess.run(self._pred, feed_dict={self._x: test_dynamic, self._v: self._template})
        attention_signals = self._sess.run(self._z, feed_dict={self._x: test_dynamic, self._v: self._template})
        return prob, attention_signals.reshape([-1, self._time_steps, 10])


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
        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)
        with tf.variable_scope(self._name):
            # self._x = tf.placeholder(tf.float32, [None, time_steps * num_features], 'input')
            self._x = tf.placeholder(tf.float32, [None, 1 * num_features], 'input')
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

    def fit(self, data_set, test_set, event_type):
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
            # self._sess.run(self._train_op,
            #                feed_dict={self._x: dynamic_feature.reshape([-1, self._time_steps * self._num_features]),
            #                           self._y: labels})
            self._sess.run(self._train_op,
                           feed_dict={self._x: np.mean(dynamic_feature, 1),
                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss_prev = loss
                loss = self._sess.run(self._loss, feed_dict={
                    # self._x: data_set.dynamic_feature.reshape([-1, self._time_steps * self._num_features]),
                    self._x: np.mean(data_set.dynamic_feature, 1),
                    self._y: data_set.labels})
                loss_diff = loss_prev - loss

                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
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
        # return self._sess.run(self._pred, feed_dict={
        #     self._x: test_set.dynamic_feature.reshape([-1, self._time_steps * self._num_features])})
        return self._sess.run(self._pred, feed_dict={
            self._x: np.mean(test_set.dynamic_feature, 1)})

    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


class CNN(object):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='CNN'):

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

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)

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
        filter_shape = [5, 100, 1, 16]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[16]), name="b")
        conv = tf.nn.conv2d(
            tf.reshape(self._x, [-1, 80, 100, 1]),
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 80 - 5 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        self._hidden_rep = tf.reshape(pooled, [-1, 16])

    def fit(self, data_set, test_set, event_type):
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
                # 由于内存不足，在case2时只能计算一个mini-batch的loss
                loss = self._sess.run(self._loss, feed_dict={self._x: dynamic_feature,
                                                             self._y: labels})
                loss_diff = loss_prev - loss

                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
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

    @property
    def name(self):
        return self._name

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


class CACNN(CNN):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 learning_rate=0.01, max_loss=0.5, max_pace=0.01, lasso=0.0, ridge=0.0,
                 optimizer=tf.train.AdamOptimizer, name='CA-CNN'):

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
        self._template_format()

        print("learning_rate=", learning_rate, "max_loss=", max_loss, "max_pace=", max_pace, "lasso=", lasso, "ridge=",
              ridge)

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')
            # self._v = tf.placeholder(tf.int32, [time_steps, 10], "context_template")

            self._sess = tf.Session()  # 会话

            self._W_trans = tf.Variable(tf.truncated_normal([1, self._num_features], stddev=0.1))

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

    def _template_format(self):
        # format the template to get contextual words
        template_i = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=np.int32).reshape([1, -1])
        # indexes of contextual words for the first word
        add_one = np.ones([10], dtype=np.int32).reshape([1, -1])
        self._template = np.zeros([0, 10], dtype=np.int32)
        for i in range(self._time_steps):
            self._template = np.concatenate([self._template, template_i], 0)
            # add the indexes to the template
            template_i = template_i + add_one
            # indexes of contextual words for the next word

    def _attention_mechanism(self):
        W_x = tf.tile(self._W_trans, [self._time_steps, 1])
        W_v = tf.tile(tf.reshape(self._W_trans, [1, -1, 1]), [self._time_steps, 1, 10])
        # weights for nonlinear mapping

        v = tf.gather(tf.pad(self._x, [[0, 0], [5, 5], [0, 0]]), self._template, axis=1)
        # get the contextual words (word embeddings)

        x_trans = tf.nn.tanh(tf.multiply(self._x, W_x))
        c_trans = tf.nn.tanh(tf.multiply(tf.transpose(v, [0, 1, 3, 2]), W_v))
        # nonlinear mapping of word and their contextual words (word embeddings). Corresponding to equation (1)and (2)

        a = tf.matmul(tf.reshape(x_trans, [-1, self._time_steps, 1, self._num_features]), c_trans)
        self._z = tf.nn.softmax(a, 3)
        # the assigned attention weights of words, denoted as Z in our paper. Corresponding to equation (3)and (4)

        u = tf.reshape(tf.matmul(self._z, v), [-1, self._time_steps, self._num_features])
        # the weighted average of word embeddings, denoted as u i our paper. Corresponding to equation (5)

        self._context = tf.contrib.layers.fully_connected(tf.concat([u, self._x], 2), self._num_features)
        # the context embeddings, denoted as c in our paper. Corresponding to equation (6)

    def _hidden_layer(self):
        filter_shape = [5, 100, 1, 16]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[16]), name="b")
        conv = tf.nn.conv2d(
            tf.reshape(self._context, [-1, 80, 100, 1]),
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 80 - 5 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        self._hidden_rep = tf.reshape(pooled, [-1, 16])

    def fit(self, data_set, test_set, event_type):
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
                # 由于内存不足，在case2时只能计算一个mini-batch的loss
                loss = self._sess.run(self._loss, feed_dict={self._x: dynamic_feature,
                                                             self._y: labels})
                loss_diff = loss_prev - loss

                y_score = self.predict(test_set)  # 此处计算和打印auc仅供调参时观察auc变化用，可删除，与最终输出并无关系
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
