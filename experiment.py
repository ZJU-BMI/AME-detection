import os
from pickle import load
from collections import Counter
import matplotlib.pyplot as plt
import time
import sklearn
import xlwt
import xlsxwriter
from imblearn.over_sampling import SMOTE
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from data import read_data, DataSet
from models import BidirectionalLSTMModel, ContextAttentionRNN, LogisticRegression, CNN, CACNN


class ExperimentSetup(object):

    kfold = 5
    batch_size = 128
    hidden_size = 128
    epochs = 20
    output_n_epochs = 1

    def __init__(self, learning_rate=0.01, max_loss=2.0, max_pace=0.01, lasso=0.0, ridge=0.0):
        self._learning_rate = learning_rate
        self._max_loss = max_loss
        self._max_pace = max_pace
        self._lasso = lasso
        self._ridge = ridge

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_loss(self):
        return self._max_loss

    @property
    def max_pace(self):
        return self._max_pace

    @property
    def lasso(self):
        return self._lasso

    @property
    def ridge(self):
        return self._ridge

    @property
    def all(self):
        return self._learning_rate, self._max_loss, self._max_pace, self._lasso, self._ridge


lr_qx_setup = ExperimentSetup(0.01, 2, 0.0001, 0, 0.001)
bi_lstm_qx_setup = ExperimentSetup(0.001, 0.5, 0.01, 0, 0.001)
ca_rnn_qx_setup = ExperimentSetup(0.001, 0.08, 0.008, 0, 0.001)

lr_xycj_setup = ExperimentSetup(0.01, 2, 0.0001, 0, 0.001)
bi_lstm_xycj_setup = ExperimentSetup(0.001, 0.5, 0.001, 0, 0.001)
ca_rnn_xycj_setup = ExperimentSetup(0.001, 0.1, 0.0025, 0, 0.001)

lr_cx_setup = ExperimentSetup(0.01, 2, 0, 0, 0.001)
bi_lstm_cx_setup = ExperimentSetup(0.001, 0.4, 0.04, 0, 0.001)
ca_rnn_cx_setup = ExperimentSetup(0.001, 0.1, 0.0025, 0, 0.001)


def evaluate(test_index, y_label, y_score, file_name):
    """
    对模型的预测性能进行评估
    :param test_index
    :param y_label: 测试样本的真实标签 true label of test-set
    :param y_score: 测试样本的预测概率 predicted probability of test-set
    :param file_name: 输出文件路径    path of output file
    """
    # TODO 全部算完再写入
    wb = xlwt.Workbook(file_name + '.xls')
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", " ", "fpr", "tpr", "thresholds", " ", "fp", "tp", "fn", "tn",
                   "fp_words", "fp_freq", "tp_words", "tp_freq", "fn_words", "fn_freq", "tn_words", "tn_freq", " ",
                   "acc", "auc", "recall", "precision", "f1-score", "threshold"]
    for i in range(len(table_title)):
        table.write(0, i, table_title[i])

    auc = roc_auc_score(y_label, y_score)

    threshold = plot_roc(y_label, y_score, table, table_title, file_name)
    y_pred_label = (y_score >= threshold) * 1
    acc = accuracy_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    f1 = f1_score(y_label, y_pred_label)

    # write metrics
    table.write(1, table_title.index("auc"), float(auc))
    table.write(1, table_title.index("acc"), float(acc))
    table.write(1, table_title.index("recall"), float(recall))
    table.write(1, table_title.index("precision"), float(precision))
    table.write(1, table_title.index("f1-score"), float(f1))

    # collect samples of FP, TP ,FN ,TN and write the result of prediction
    fp_sentences = []
    fn_sentences = []
    tp_sentences = []
    tn_sentences = []
    fp_count = 1
    tp_count = 1
    fn_count = 1
    tn_count = 1
    # sentence_set = load(open("resources/all_sentence_progress_note1.pkl", "rb"))
    sentence_set = load(open("resources/all_sentences_admission_records.pkl", "rb"))
    for j in range(len(y_label)):
        if y_label[j] == 0 and y_pred_label[j] == 1:  # FP
            write_result(j, test_index, y_label, y_score, y_pred_label, table, table_title, sentence_set, fp_sentences,
                         "fp", fp_count)
            fp_count += 1
        if y_label[j] == 1 and y_pred_label[j] == 1:  # TP
            write_result(j, test_index, y_label, y_score, y_pred_label, table, table_title, sentence_set, tp_sentences,
                         "tp", tp_count)
            tp_count += 1
        if y_label[j] == 1 and y_pred_label[j] == 0:  # FN
            write_result(j, test_index, y_label, y_score, y_pred_label, table, table_title, sentence_set, fn_sentences,
                         "fn", fn_count)
            fn_count += 1
        if y_label[j] == 0 and y_pred_label[j] == 0:  # TN
            write_result(j, test_index, y_label, y_score, y_pred_label, table, table_title, sentence_set, tn_sentences,
                         "tn", tn_count)
            tn_count += 1

    # word frequency statistic
    write_word_frequency(fp_sentences, table, table_title, "fp")
    write_word_frequency(tp_sentences, table, table_title, "tp")
    write_word_frequency(fn_sentences, table, table_title, "fn")
    write_word_frequency(tn_sentences, table, table_title, "tn")

    wb.save(file_name + ".xls")


def write_result(j, index, y_label, y_score, y_pred_label, table, table_title, sentence_set, samples, group_name,
                 count):
    """
    1.write the indexs of test samples and its true label, predicted probabilities and predicted labels
    2.collect samples of FP, TP ,FN ,TN
    :param j:
    :param index:
    :param y_label:
    :param y_score:
    :param y_pred_label:
    :param table:
    :param table_title:
    :param sentence_set:
    :param samples:
    :param group_name:
    :param count:
    :return:
    """
    table.write(j + 1, table_title.index("test_index"), int(index[j]))
    table.write(j + 1, table_title.index("label"), int(y_label[j]))
    table.write(j + 1, table_title.index("prob"), float(y_score[j]))
    table.write(j + 1, table_title.index("pre"), int(y_pred_label[j]))
    samples.extend(sentence_set[index[j]])
    table.write(count, table_title.index(group_name), int(index[j]))


def write_word_frequency(samples, table, table_title, group_name):
    # TODO 命名需改
    """
    词频统计并写入xls文件
    :param samples: group of collected samples(samples of FP, TP, FN ,TN)
    :param table: the sheet of workbook
    :param table_title: the first row of table
    :param group_name: name of group
    :return:
    """
    words = Counter(samples).most_common()

    for i, word_with_freq in enumerate(words):
        word, freq = word_with_freq
        table.write(i + 1, table_title.index(group_name + "_words"), word)
        table.write(i + 1, table_title.index(group_name + "_freq"), freq)


def plot_roc(test_labels, test_predictions, table, table_title, filename):
    """
    1.plot and save the  ROC curve with AUC value
    2.record the FPR, TPR and thresholds
    3.choose the threshold with max(TPR-FPR)
    :param test_labels: 测试集标签
    :param test_predictions: 测试集预测值
    :param table: xls文件的sheet
    :param table_title: 表头字符串数组
    :param filename: 图片文件名
    :return: optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    table.write(2, table_title.index("threshold"), float(threshold))
    auc = "%.3f" % sklearn.metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = ' + str(auc)
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.savefig(filename + '.png', format='png')
        plt.close()
    return threshold


def imbalance_preprocess(train_dynamic, train_y):  # SMOTE过采样
    """
    SMOTE处理不平衡数据集
    :param train_dynamic: 训练集输入
    :param train_y: 训练集标签
    :return: 处理后的训练集输入和标签
    """
    method = SMOTE(kind='regular')
    x_res, y_res = method.fit_sample(train_dynamic.reshape([-1, 100 * train_dynamic.shape[1]]), train_y[:, -1])
    train_dynamic_res = x_res.reshape([-1, train_dynamic.shape[1], 100])
    train_y_res = y_res.reshape([-1, 1])
    return train_dynamic_res, train_y_res


class LogisticRegressionExperiment(object):
    def __init__(self, event_type):
        self._event_type = event_type
        self._data_set = read_data(event_type)
        self._num_features = self._data_set.dynamic_feature.shape[2]
        self._time_steps = self._data_set.dynamic_feature.shape[1]
        self._n_output = self._data_set.labels.shape[1]
        print(event_type)
        self._model_format()
        self._check_path()

    def _model_format(self):
        if self._event_type == "qx":
            learning_rate, max_loss, max_pace, lasso, ridge = lr_qx_setup.all
        elif self._event_type == "cx":
            learning_rate, max_loss, max_pace, lasso, ridge = lr_cx_setup.all
        else:
            learning_rate, max_loss, max_pace, lasso, ridge = lr_xycj_setup.all
        self._model = LogisticRegression(num_features=self._num_features,
                                         time_steps=self._time_steps,
                                         n_output=self._n_output,
                                         batch_size=ExperimentSetup.batch_size,
                                         epochs=ExperimentSetup.epochs,
                                         output_n_epoch=ExperimentSetup.output_n_epochs,
                                         learning_rate=learning_rate,
                                         max_loss=max_loss,
                                         max_pace=max_pace,
                                         lasso=lasso,
                                         ridge=ridge)

    def _check_path(self):
        if not os.path.exists("average_result_cx_TEST" + self._event_type):
            os.makedirs("average_result_cx_TEST" + self._event_type)
        self._filename = "average_result_cx_TEST" + self._event_type + "/" + self._model.name + " " + time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime())

    def do_experiments(self):
        dynamic_feature = self._data_set.dynamic_feature
        labels = self._data_set.labels
        kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=False)

        n_output = labels.shape[1]  # classes

        tol_test_index = np.zeros(shape=0, dtype=np.int32)
        tol_pred = np.zeros(shape=(0, n_output))
        tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)
        i = 1
        for train_idx, test_idx in kf.split(X=dynamic_feature, y=labels.reshape(-1)):  # 五折交叉
            train_dynamic = dynamic_feature[train_idx]
            train_y = labels[train_idx]
            train_dynamic_res, train_y_res = imbalance_preprocess(train_dynamic, train_y)  # SMOTE过采样方法处理不平衡数据集

            test_dynamic = dynamic_feature[test_idx]
            test_y = labels[test_idx]

            train_set = DataSet(train_dynamic_res, train_y_res)
            test_set = DataSet(test_dynamic, test_y)

            self._model.fit(train_set, test_set, self._event_type)

            y_score = self._model.predict(test_set)

            tol_test_index = np.concatenate((tol_test_index, test_idx))
            tol_pred = np.vstack((tol_pred, y_score))
            tol_label = np.vstack((tol_label, test_y))
            print("Cross validation: {} of {}".format(i, ExperimentSetup.kfold),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            i += 1
        evaluate(tol_test_index, tol_label, tol_pred, self._filename)
        self._model.close()


class BidirectionalLSTMExperiments(LogisticRegressionExperiment):
    def __init__(self, event_type):
        super().__init__(event_type)

    def _model_format(self):
        if self._event_type == "qx":
            learning_rate, max_loss, max_pace, lasso, ridge = bi_lstm_qx_setup.all
        elif self._event_type == "cx":
            learning_rate, max_loss, max_pace, lasso, ridge = bi_lstm_cx_setup.all
        else:
            learning_rate, max_loss, max_pace, lasso, ridge = bi_lstm_xycj_setup.all
        self._model = BidirectionalLSTMModel(num_features=self._num_features,
                                             time_steps=self._time_steps,
                                             lstm_size=ExperimentSetup.hidden_size,
                                             n_output=self._n_output,
                                             batch_size=ExperimentSetup.batch_size,
                                             epochs=ExperimentSetup.epochs,
                                             output_n_epoch=ExperimentSetup.output_n_epochs,
                                             learning_rate=learning_rate,
                                             max_loss=max_loss,
                                             max_pace=max_pace,
                                             lasso=lasso,
                                             ridge=ridge)


class ContextAttentionRNNExperiments(LogisticRegressionExperiment):
    def __init__(self, event_type):
        super().__init__(event_type)

    def _model_format(self):
        if self._event_type == "qx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_qx_setup.all
        elif self._event_type == "cx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_cx_setup.all
        else:
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_xycj_setup.all
        self._model = ContextAttentionRNN(num_features=self._num_features,
                                          time_steps=self._time_steps,
                                          lstm_size=ExperimentSetup.hidden_size,
                                          n_output=self._n_output,
                                          batch_size=ExperimentSetup.batch_size,
                                          epochs=ExperimentSetup.epochs,
                                          output_n_epoch=ExperimentSetup.output_n_epochs,
                                          learning_rate=learning_rate,
                                          max_loss=max_loss,
                                          max_pace=max_pace,
                                          lasso=lasso,
                                          ridge=ridge)

    def attention_analysis(self):
        dynamic_feature = self._data_set.dynamic_feature
        labels = self._data_set.labels
        kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=True)
        # models = ["qx_case1_save_net06-20-10-54.ckpt", "qx_case1_save_net06-20-10-58.ckpt",
        #           "qx_case1_save_net06-20-11-03.ckpt", "qx_case1_save_net06-20-11-08.ckpt",
        #           "qx_case1_save_net06-20-11-13.ckpt"]
        # models = ["xycj_case1_save_net06-20-11-45.ckpt", "xycj_case1_save_net06-20-11-51.ckpt",
        #           "xycj_case1_save_net06-20-11-56.ckpt", "xycj_case1_save_net06-20-12-01.ckpt",
        #           "xycj_case1_save_net06-20-12-06.ckpt"]
        models = ["cx_case1_save_net06-20-11-19.ckpt", "cx_case1_save_net06-20-11-24.ckpt",
                  "cx_case1_save_net06-20-11-29.ckpt", "cx_case1_save_net06-20-11-35.ckpt",
                  "cx_case1_save_net06-20-11-40.ckpt"]
        i_fold = 1
        test_idx_tol = np.zeros(0, dtype=np.int32)
        attention_signals_tol = np.zeros(shape=(0, 80, 10))
        test_labels_tol = np.zeros([0, 1])
        pre_tol = np.zeros([0, 1])
        for train_idx, test_idx in kf.split(X=dynamic_feature, y=labels.reshape(-1)):  # 五折交叉
            test_dynamic = dynamic_feature[test_idx]
            test_labels = labels[test_idx]
            prob, attention_signals = self._model.attention_analysis(test_dynamic, models[i_fold - 1])
            fpr, tpr, thresholds = roc_curve(test_labels, prob, pos_label=1)
            threshold = thresholds[np.argmax(tpr - fpr)]
            pre = (prob >= threshold) * 1
            attention_signals_tol = np.concatenate((attention_signals_tol, attention_signals))
            test_idx_tol = np.concatenate((test_idx_tol, test_idx))
            test_labels_tol = np.concatenate((test_labels_tol, test_labels))
            pre_tol = np.concatenate((pre_tol, pre))
            i_fold += 1
        # break
        attention_write(attention_signals_tol, test_idx_tol, dynamic_feature.shape[1], test_labels_tol, pre_tol)
        self._model.close()


def dict_form(sentence_set, test_index, sample, attention_signals_avg, dict):
    # dict_temp = {}
    # for step, word in enumerate(sentence_set[test_index[sample]]):
    #     if word in dict_temp:
    #         dict_temp[word].append(attention_signals_avg[sample, step])
    #     else:
    #         dict_temp[word] = [attention_signals_avg[sample, step]]
    # for (d, v) in dict_temp.items():
    #     if d in dict:
    #         dict[d].append(np.sum(dict_temp[d]))
    #     else:
    #         dict[d] = [np.sum(dict_temp[d])]
    for step, word in enumerate(sentence_set[test_index[sample]]):
        if word in dict:
            dict[word].append(attention_signals_avg[sample, step])
        else:
            dict[word] = [attention_signals_avg[sample, step]]


def attention_write(attention_signals, test_index, n_steps, labels, pre):
    attention_signals_shift = np.zeros([len(test_index), n_steps, n_steps + 10])
    for i_sample in range(len(test_index)):
        for j in range(n_steps):
            attention_signals_shift[i_sample, j, j:j + 5] = attention_signals[i_sample, j, 0:5]
            attention_signals_shift[i_sample, j, j + 6:j + 11] = attention_signals[i_sample, j, 5:10]
    attention_signals_avg = np.sum(attention_signals_shift, 1)[:, 5:n_steps + 5] / 10
    # sentence_set = load(open("resources/all_sentences_progress_notes.pkl", "rb"))
    sentence_set = load(open("resources/all_sentences_admission_records.pkl", "rb"))
    workbook = xlsxwriter.Workbook("cx_case1_attention_all_average.xlsx")
    table = workbook.add_worksheet()
    dict_fp = {}
    dict_tp = {}
    dict_fn = {}
    dict_tn = {}
    for sample in range(len(test_index)):
        if labels[sample] == 0 and pre[sample] == 1:
            dict_form(sentence_set, test_index, sample, attention_signals_avg, dict_fp)
        if labels[sample] == 1 and pre[sample] == 1:
            dict_form(sentence_set, test_index, sample, attention_signals_avg, dict_tp)
        if labels[sample] == 1 and pre[sample] == 0:
            dict_form(sentence_set, test_index, sample, attention_signals_avg, dict_fn)
        if labels[sample] == 0 and pre[sample] == 0:
            dict_form(sentence_set, test_index, sample, attention_signals_avg, dict_tn)

    for (d, v) in dict_fp.items():
        dict_fp[d] = np.average(dict_fp[d])
    dict_fp = sorted(dict_fp.items(), key=lambda d: d[1], reverse=True)
    for (d, v) in dict_tp.items():
        dict_tp[d] = np.average(dict_tp[d])
    dict_tp = sorted(dict_tp.items(), key=lambda d: d[1], reverse=True)
    for (d, v) in dict_fn.items():
        dict_fn[d] = np.average(dict_fn[d])
    dict_fn = sorted(dict_fn.items(), key=lambda d: d[1], reverse=True)
    for (d, v) in dict_tn.items():
        dict_tn[d] = np.average(dict_tn[d])
    dict_tn = sorted(dict_tn.items(), key=lambda d: d[1], reverse=True)

    for i, d in enumerate(dict_fp):
        table.write(i + 1, 1, d[0])
        table.write(i + 1, 2, d[1])
    for i, d in enumerate(dict_tp):
        table.write(i + 1, 4, d[0])
        table.write(i + 1, 5, d[1])
    for i, d in enumerate(dict_fn):
        table.write(i + 1, 7, d[0])
        table.write(i + 1, 8, d[1])
    for i, d in enumerate(dict_tn):
        table.write(i + 1, 10, d[0])
        table.write(i + 1, 11, d[1])
    workbook.close()


class CNNExperiments(LogisticRegressionExperiment):
    def __init__(self, event_type):
        super().__init__(event_type)

    def _model_format(self):
        if self._event_type == "qx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_qx_setup.all
        elif self._event_type == "cx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_cx_setup.all
        else:
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_xycj_setup.all
        self._model = CNN(num_features=self._num_features,
                          time_steps=self._time_steps,
                          lstm_size=ExperimentSetup.hidden_size,
                          n_output=self._n_output,
                          batch_size=ExperimentSetup.batch_size,
                          epochs=ExperimentSetup.epochs,
                          output_n_epoch=ExperimentSetup.output_n_epochs,
                          learning_rate=learning_rate,
                          max_loss=max_loss,
                          max_pace=max_pace,
                          lasso=lasso,
                          ridge=ridge)


class CACNNExperiments(LogisticRegressionExperiment):
    def __init__(self, event_type):
        super().__init__(event_type)

    def _model_format(self):
        if self._event_type == "qx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_qx_setup.all
        elif self._event_type == "cx":
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_cx_setup.all
        else:
            learning_rate, max_loss, max_pace, lasso, ridge = ca_rnn_xycj_setup.all
        self._model = CACNN(num_features=self._num_features,
                            time_steps=self._time_steps,
                            lstm_size=ExperimentSetup.hidden_size,
                            n_output=self._n_output,
                            batch_size=ExperimentSetup.batch_size,
                            epochs=ExperimentSetup.epochs,
                            output_n_epoch=ExperimentSetup.output_n_epochs,
                            learning_rate=learning_rate,
                            max_loss=max_loss,
                            max_pace=max_pace,
                            lasso=lasso,
                            ridge=ridge)


if __name__ == '__main__':
    # TODO此处字符串改为constant
    ischemia = "qx"
    bleeding = "cx"
    revascularization = "xycj"

    for i in range(5):
        # LogisticRegressionExperiment(ischemia).do_experiments()
        # MultiLayerPercptronExperimrnt(ischemia).do_experiments()
        #
        # LogisticRegressionExperiment(bleeding).do_experiments()
        # MultiLayerPercptronExperimrnt(bleeding).do_experiments()
        #
        # LogisticRegressionExperiment(revascularization).do_experiments()
        # MultiLayerPercptronExperimrnt(revascularization).do_experiments()

        ContextAttentionRNNExperiments(ischemia).do_experiments()
        # BidirectionalLSTMExperiments(ischemia).do_experiments()

        ContextAttentionRNNExperiments(bleeding).do_experiments()
        # BidirectionalLSTMExperiments(bleeding).do_experiments()

        ContextAttentionRNNExperiments(revascularization).do_experiments()
        # BidirectionalLSTMExperiments(revascularization).do_experiments()

        # ContextAttentionRNNWithOriginExperiments(revascularization).do_experiments()
        # ContextAttentionRNNWithOriginExperiments(bleeding).do_experiments()

        # CNNExperiments(ischemia).do_experiments()
        # CNNExperiments(revascularization).do_experiments()
        # CNNExperiments(bleeding).do_experiments()
        #
        # CACNNExperiments(ischemia).do_experiments()
        # CACNNExperiments(revascularization).do_experiments()
        # CACNNExperiments(bleeding).do_experiments()
