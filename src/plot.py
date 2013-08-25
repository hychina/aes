import csv
import pylab as py
from pylab import *
from main import *
from util import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor

def score_count_plot():
    path = '../data/new_score/new_train.tsv'
    r = csv.reader(open(path), delimiter = '\t')
    score_count = dict()
    for row in r:
        # if row[1] == '9':
            if row[2] in score_count:
                score_count[row[2]] += 1
            else:
                score_count[row[2]] = 1

    X = []
    Y = []
    for score in score_count:
        X.append(int(score))
        Y.append(score_count[score])

    print X, Y
    bar(X, Y)
    save

def feature_score_plot():
    path = "../data/features/all_train.txt"
    r = csv.reader(open(path), delimiter = '\t')
    features = next(r)[1:] 
    rows = []
    for row in r:
        rows.append(row)

    scores = []
    path = '../data/new_score/new_train.tsv'
    r = csv.reader(open(path), delimiter = '\t')
    for row in r:
        scores.append(row)

    num_sets = get_essay_set_range("train").keys()
    for set_no in num_sets:
        set_range = get_essay_set_range("train")[set_no]
        start = int(set_range[0])
        end = int(set_range[1])

        y = []
        for row in scores:
            essay_id = int(row[0])
            if essay_id >= start and essay_id <= end:
                y.append(int(row[2]))

        for n, f in enumerate(features):
            print n, f
            X = []
            for row in rows:
                essay_id = int(row[0])
                if essay_id >= start and essay_id <= end:
                    X.append(float(row[n + 1]))

            close()
            scatter(X, y)
            savefig("../data/plots/" + 
                    str(n) + 
                    "-" + set_no +
                    "-" + str(f) + ".png",
                    bbox_inches=0)

def feature_importance():
    trainset = load_dataset("train")

    forest = ExtraTreesClassifier(n_estimators = 100,
                                  compute_importances = True,
                                  random_state = 0)
    forest.fit(trainset.data, trainset.score)

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    importances = forest.feature_importances_
    print std
    print importances

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print "Feature ranking:"

    for f in xrange(trainset.n_features):
        print "%d. %s (%f)" % (f + 1, trainset.feature_names[indices[f]], importances[indices[f]])

    # Plot the feature importances of the forest
    pl.figure()
    pl.title("Feature importances")
    pl.bar(xrange(trainset.n_features), importances[indices],
           color="r", yerr=std[indices], align="center")
    pl.xticks(xrange(trainset.n_features), indices)
    pl.xlim([-1, trainset.n_features])
    pl.show()

feature_score_plot()
# feature_importance()
