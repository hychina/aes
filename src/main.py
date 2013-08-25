# coding=utf-8

import csv
import pickle
import myeval
import util
import numpy as np
import pylab as pl
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def get_num_lines(filename):
    tmp = open(filename, 'r')
    num_lines = 0
    line = tmp.readline()
    while line:
        num_lines += 1
        line = tmp.readline()
    tmp.close()
    return num_lines

def load_dataset(category):
    
    feature_file = ""
    score_file = ""

    if category == "train":
        feature_file = "../data/features/all_train.txt"
        score_file = "../data/new_score/new_train.tsv"
    elif category == "valid":
        feature_file = "../data/features/all_valid.txt"
        score_file = "../data/new_score/new_valid.tsv"

    features = csv.reader(open(feature_file, 'r'), delimiter = '\t')
    scores = csv.reader(open(score_file, 'r'), delimiter = '\t')
    feature_names = next(features)[1:]
    n_samples = get_num_lines(score_file)
    n_features = len(feature_names)

    data = np.empty((n_samples, n_features))
    score = np.empty((n_samples,))

    for i, row in enumerate(features):
        data[i] = np.asarray(row[1:])

    for i, row in enumerate(scores):
        score[i] = row[2]

    return Bunch(data = data, 
                 score = score, 
                 n_samples = n_samples, 
                 n_features = n_features, 
                 feature_names = feature_names) 

def select_feature(dataset, features):
    if features == "all":
        return dataset.data

    cols = []
    for f in features:
        cols.append(dataset.feature_names.index(f))

    n_features = len(features)
    new_data = np.empty((dataset.n_samples, n_features))
    for i, row in enumerate(dataset.data):
        new_row = []
        for c in cols:
            new_row.append(row[c])
        new_data[i] = np.asarray(new_row)
    return new_data

def numbers_in_range(begin, end, step):
    numbers = []
    n = begin
    while n <= end:
        numbers.append(n)
        n += step
    return np.array(numbers)

def get_test_result(real_scores, predicted_scores):
    result = []
    for score in real_scores:
        result.append(['1', int(score)])
    for i, score in enumerate(predicted_scores):
        result[i].append(int(score))
    f = open('result.tsv', 'w')
    w = csv.writer(f, delimiter = '\t')
    w.writerows(result)
    f.close()
    return myeval.evaluate("result.tsv")

def valid(selection):
    trainset = load_dataset("train")
    validset = load_dataset("valid")
    trainset.data = select_feature(trainset, selection)
    validset.data = select_feature(validset, selection)

    # 在训练集上进行交叉验证
    X = trainset.data
    y = trainset.score
    # split the training set into ten folds
    kf = cross_validation.KFold(len(y), n_folds = 10)

    highest = 0
    best_alpha = 0

    # 通过交叉验证选择模型参数
    for alpha in numbers_in_range(0.1, 2, 0.1):

        test_results = []
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = linear_model.Ridge(alpha = alpha)
            clf.fit(X_train, y_train)
            predicted_scores = clf.predict(X_test)
            test_results.append(get_test_result(y_test, predicted_scores))

        mean = np.mean(test_results)
        print alpha, mean

        if mean > highest:
            highest = mean
            best_alpha = alpha

    print best_alpha, highest

    # 在测试集上的得分
    clf = linear_model.Ridge(alpha = best_alpha)
    clf.fit(trainset.data, trainset.score)
    predicted_scores = clf.predict(validset.data)
    print "在validset上的得分"
    validset_score = get_test_result(validset.score, predicted_scores)
    print ""

    # 返回交叉验证的结果
    return (best_alpha, highest, validset_score)

# 用贪心法选择特征
def evaluate_features(features):
    result_file = open('feature_selection_result.txt', 'w')
    selected_features = features
    highest_score = 0
    best_feature = ''

    while True:
        for f in selected_features:
            tmp = list()
            for feature in selected_features:
                if feature != f:
                    tmp.append(feature)
            # 进行交叉验证
            best_alpha, score, validset_score = valid(tmp)
            result_file.write('removed feature: ' + f + '\n'
                              'alpha: ' + str(best_alpha) + '\n'
                              'cv score: ' + str(score) + '\n'
                              'validset score: ' + str(validset_score) + '\n\n')
            result_file.flush()

            if score > highest_score:
                highest_score = score
                best_feature = f
        if best_feature != '':
            selected_features.remove(best_feature)
            result_file.write('\n已经将特征' + best_feature + '删除\n')
            best_feature = ''
        else:
            break

    result_str = '\n最终得分: ' + str(highest_score) + '\n' + \
                  '所选特征: ' + selected_features + '\n' + \
                  '所用参数: ' + str(best_alpha)
    result_file.write(result_str)
    result_file.close()

def evaluate_each_feature(features):
    out = open('per_feature.txt', 'w')
    feature_score_list = list()
    for f in features:
        best_alpha, cv_highest, validset_score = valid([f])
        i = len(feature_score_list) - 1
        insert_position = i + 1
        while i >= 0:
            if feature_score_list[i][1] < cv_highest:
                insert_position = i
                i -= 1
            else:
                insert_position = i + 1
                break
        feature_score_list.insert(insert_position, (f, cv_highest, validset_score))

    for feature_score in feature_score_list:
        out.write(feature_score[0] + ':\n' + 
                  '交叉验证最高得分: ' + str(feature_score[1]) + '\n' +
                  '测试集最高得分: ' + str(feature_score[2]) + '\n\n')

# 在ValidSet进行贪心选择的结果
selected_features = ['tree_uh', 't_s', 'lWdCnt', 'tree_nodeCount', 
                     'tree_jrbrs', 'spelling_error', 'mlt', 'avgDpt', 'tree_wh', 
                     'gErrTypes', 'vp_t', 
                     'wdCnt', 'dptSD', 'ttr2', 'readability1', 'readability2', 
                     'ct_t', 'readability5', 'readability4', 'verbCnt', 'tree_cc', 
                     'verbal_phrases', 'tree_depth', 'adjCnt', 'vbgCnt', 
                     'tree_to', 'cp_t', 'cp_c', 'tree_npvp', 
                     'dc_c', 'lexical_diversity', 'nunCnt', 'gErrCnt', 
                     'readability3', 'ttr1', 'c_s', 'gErrDen', 
                     'nbDptSD', 'tree_md', 'tree_grammerErr', 'cn_t']

# 所有特征
all_features = ["tree_uh", "word_level", "wdCnt", "nunCnt", "lexical_diversity",
            "lWdCnt", "StsCnt", "verbCnt", "ttr1", "punCnt",
            "adjCnt", "tree_npvp", "tree_depth", "tree_to", "tree_md",
            "cmaCnt", "ttr2", "readability5", "tree_grammerErr", "tree_jrbrs",
            "tree_cc", "tree_nodeCount", "avgGErrSpan", "nbDptSD", "tree_wh",
            "mlc", "sErrCnt", "sErrTypes", "readability3", "readability1",
            "advCnt", "avgDpt", "mlt", "gErrCnt", "vbgCnt",
            "gErrTypes", "mls", "dc_t", "ct_t", "c_s",
            "gErrSpanSum", "spelling_error", "dc_c", "readability2", "gErrDen",
            "dptSD", "c_t", "vp_t", "wdLen", "cn_c",
            "verbal_phrases", "nbDptD", "cn_t", "cp_t", "readability4",
            "cp_c", "t_s"] 

# 选出单个特征得分高于0.1的
selected_features_2 = ["word_level", "tree_to", "ttr1", "tree_cc", "ttr2", 
                       "lWdCnt", "lexical_diversity", "tree_wh", "tree_md", "tree_grammerErr", 
                       "tree_npvp", "wdCnt", "nunCnt", "cmaCnt", "adjCnt", 
                       "tree_jrbrs", "punCnt", "StsCnt"]

# evaluate_features(all_features)
# evaluate_each_feature(all_features)
