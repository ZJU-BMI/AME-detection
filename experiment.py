import os
from pickle import load
from collections import Counter
import matplotlib.pyplot as plt
import time
import sklearn
import xlwt
from imblearn.over_sampling import SMOTE
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve  # roc计算曲线
from data import read_data, DataSet
from models import BidirectionalLSTMModel, CA_RNN, LogisticRegression, MultiLayerPerceptron


class ExperimentSetup(object):
    kfold = 5
    batch_size = 64

    lstm_size = 128
    learning_rate = 0.0001
    epochs = 20
    output_n_epochs = 1


def evaluate(test_index, y_label, y_score, file_name):
    """
    对模型的预测性能进行评估
    :param tol_label: 测试样本的真实标签
    :param tol_pred: 测试样本的预测概率分布
    :param event_type: 预测事件类型  target type of AME
    :param model_name: 使用的模型    used model
    :return: 正确率，AUC，精度，召回率， F1值
    """
    # TODO: 输出文件已添加测试集下标，实际和预测值，需加入TP，FP统计及其词频统计
    wb = xlwt.Workbook(file_name + '.xls')
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", "fpr", "tpr", "thresholds", "fp", "tp", "fn", "tn", "fp_words",
                   "fp_freq", "tp_words", "tp_freq", "fn_words", "fn_freq", "tn_words", "tn_freq", "acc", "auc",
                   "recall", "precision", "f1-score", "threshold"]
    for i in range(len(table_title)):
        table.write(0, i, table_title[i])

    auc = roc_auc_score(y_label, y_score)

    threshold = plot_roc(y_label, y_score, table, table_title, file_name)
    y_pred_label = (y_score >= threshold)
    acc = accuracy_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    f1 = f1_score(y_label, y_pred_label)

    table.write(1, table_title.index("auc"), float(auc))
    table.write(1, table_title.index("acc"), float(acc))
    table.write(1, table_title.index("recall"), float(recall))
    table.write(1, table_title.index("precision"), float(precision))
    table.write(1, table_title.index("f1-score"), float(f1))

    fp = []
    tp = []
    fn = []
    tn = []
    for j in range(len(y_label)):
        if y_label[j] == 0 and y_pred_label[j] == 1:  # FP
            fp.append(test_index[j])
        if y_label[j] == 1 and y_pred_label[j] == 1:  # TP
            tp.append(test_index[j])
        if y_label[j] == 1 and y_pred_label[j] == 0:  # FN
            fn.append(test_index[j])
        if y_label[j] == 0 and y_pred_label[j] == 0:  # TN
            tn.append(test_index[j])

        table.write(j + 1, table_title.index("test_index"), int(test_index[j]))
        table.write(j + 1, table_title.index("label"), int(y_label[j]))
        table.write(j + 1, table_title.index("prob"), float(y_score[j]))
        table.write(j + 1, table_title.index("pre"), int(y_pred_label[j]))

    fp_sentence = []
    fn_sentence = []
    tp_sentence = []
    tn_sentence = []
    sentence_set = load(open("all_sentences.pkl", "rb"))
    for i in range(len(fp)):
        fp_sentence.extend(sentence_set[fp[i]])
        table.write(i + 1, table_title.index("fp"), int(fp[i]))
    for i in range(len(fn)):
        fn_sentence.extend(sentence_set[fn[i]])
        table.write(i + 1, table_title.index("fn"), int(fn[i]))
    for i in range(len(tp)):
        tp_sentence.extend(sentence_set[tp[i]])
        table.write(i + 1, table_title.index("tp"), int(tp[i]))
    for i in range(len(tn)):
        tn_sentence.extend(sentence_set[tn[i]])
        table.write(i + 1, table_title.index("tn"), int(tn[i]))
    fp_words = Counter(fp_sentence).most_common()
    fn_words = Counter(fn_sentence).most_common()
    tp_words = Counter(tp_sentence).most_common()
    tn_words = Counter(tn_sentence).most_common()
    j = 1
    for i in fp_words:
        (a, b) = i
        table.write(j, table_title.index("fp_words"), a)
        table.write(j, table_title.index("fp_freq"), b)
        j = j + 1
    j = 1
    for i in fn_words:
        (a, b) = i
        table.write(j, table_title.index("fn_words"), a)
        table.write(j, table_title.index("fn_freq"), b)
        j = j + 1
    j = 1
    for i in tp_words:
        (a, b) = i
        table.write(j, table_title.index("tp_words"), a)
        table.write(j, table_title.index("tp_freq"), b)
        j = j + 1
    j = 1
    for i in tn_words:
        (a, b) = i
        table.write(j, table_title.index("tn_words"), a)
        table.write(j, table_title.index("tn_freq"), b)
        j = j + 1
    wb.save(file_name + ".xls")
    return acc, auc, precision, recall, f1


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
    fpr, tpr, thresholds = roc_curve(
        test_labels, test_predictions, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    auc = "%.3f" % sklearn.metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = ' + str(auc)
    with plt.style.context(('ggplot')):
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
    return threshold


def model_experiments(model, data_set, filename):
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels
    kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=True)

    n_output = labels.shape[1]  # classes

    tol_test_index = np.zeros(shape=0, dtype=np.int32)
    tol_pred = np.zeros(shape=(0, n_output))
    tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)
    i = 1
    for train_idx, test_idx in kf.split(X=data_set.dynamic_feature, y=data_set.labels):  # 五折交叉
        train_dynamic = dynamic_feature[train_idx]
        train_y = labels[train_idx]

        train_dynamic_res, train_y_res = imbalance_preprocess(train_dynamic, train_y)  # SMOTE过采样方法处理不平衡数据集
        train_set = DataSet(train_dynamic_res, train_y_res)

        model.fit(train_set)

        test_dynamic = dynamic_feature[test_idx]
        test_y = labels[test_idx]
        test_set = DataSet(test_dynamic, test_y)

        y_score = model.predict(test_set)

        tol_test_index = np.concatenate((tol_test_index, test_idx))
        tol_pred = np.vstack((tol_pred, y_score))
        tol_label = np.vstack((tol_label, test_y))
        print("Cross validation: {} of {}".format(i, ExperimentSetup.kfold))
        i += 1
    return evaluate(tol_test_index, tol_label, tol_pred, filename)


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


def bidirectional_lstm_model_experiments(event_type):
    data_set = read_data()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = BidirectionalLSTMModel(num_features,
                                   time_steps,
                                   ExperimentSetup.lstm_size,
                                   n_output,
                                   batch_size=ExperimentSetup.batch_size,
                                   epochs=ExperimentSetup.epochs,
                                   output_n_epoch=ExperimentSetup.output_n_epochs)
    if not os.path.exists("result_" + event_type):
        os.makedirs("result_" + event_type)
    filename = "result_" + event_type + "/Bi-LSTM " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return model_experiments(model, data_set, filename)


def CA_RNN_experiments(event_type):
    data_set = read_data()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = CA_RNN(num_features,
                   time_steps,
                   ExperimentSetup.lstm_size,
                   n_output,
                   batch_size=ExperimentSetup.batch_size,
                   epochs=ExperimentSetup.epochs,
                   output_n_epoch=ExperimentSetup.output_n_epochs)

    if not os.path.exists("result_" + event_type):
        os.makedirs("result_" + event_type)
    filename = "result_" + event_type + "/CA-RNN " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return model_experiments(model, data_set, filename)


def logistic_regression_experiments(event_type):
    data_set = read_data()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = LogisticRegression(num_features,
                               time_steps,
                               n_output,
                               batch_size=ExperimentSetup.batch_size,
                               epochs=ExperimentSetup.epochs,
                               output_n_epoch=ExperimentSetup.output_n_epochs)

    if not os.path.exists("result_" + event_type):
        os.makedirs("result_" + event_type)
    filename = "result_" + event_type + "/LR " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return model_experiments(model, data_set, filename)


def multi_layer_perceptron_experiments(event_type):
    data_set = read_data()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = MultiLayerPerceptron(num_features,
                                 time_steps,
                                 hidden_units=ExperimentSetup.lstm_size,
                                 n_output=n_output,
                                 batch_size=ExperimentSetup.batch_size,
                                 epochs=ExperimentSetup.epochs,
                                 output_n_epoch=ExperimentSetup.output_n_epochs)

    if not os.path.exists("result_" + event_type):
        os.makedirs("result_" + event_type)
    filename = "result_" + event_type + "/MLP " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return model_experiments(model, data_set, filename)


if __name__ == '__main__':
    # bidirectional_lstm_model_experiments('qx')
    # CA_RNN_experiments('qx')
    # logistic_regression_experiments("qx")
    multi_layer_perceptron_experiments("qx")
