# coding=utf-8

import os
from os.path import join
import csv

path = '../data/features/'

def prep_train():
    file_names = []
    for f in os.listdir(path):
        if f.startswith("train_") and \
           f != 'train_lexical_complexity.txt' and \
           f != 'train_readability.txt':
            file_names.append(join(path,f))

    matrix = []
    header = []
    matrix.append(header)
    
    for i, f in enumerate(file_names):
        feature_name = f[len(path) + len("train_"):-4]

        # 判断delimiter
        delimiter = ''
        tmp = open(f, 'r')
        splits = len(next(tmp).split(' '))
        if splits > 1:
            delimiter = ' '
        else:
            delimiter = '\t'
        tmp.close()

        r = csv.reader(open(f, 'r'), delimiter = delimiter)

        if i == 0:
            header.append('essay_id')
            header.append(feature_name)
            for row in r:
                matrix.append(row)
        else:
            header.append(feature_name)
            for n, row in enumerate(r):
                matrix[n + 1].append(row[1])

    # 单独处理train_lexical_complexity.txt
    r = csv.reader(open(join(path, "train_lexical_complexity.txt"), 'r'), delimiter = '\t')
    feature_names = next(r)[1:]
    header.extend(feature_names)
    for n, row in enumerate(r):
        matrix[n + 1].extend(row[1:-1]) # 最后一列多出一个制表符

    # 单独处理train_readability.txt
    r = csv.reader(open(join(path, "train_readability.txt"), 'r'), delimiter = '\t')
    for i in range(1, 6):
        header.append("readability" + str(i))
    for n, row in enumerate(r):
        matrix[n + 1].extend(row[1:])

    w = csv.writer(open('../data/features/all_train.txt', 'w'), delimiter = '\t')
    w.writerows(matrix)
    
def prep_valid():
    file_names = []
    for f in os.listdir(path):
        if f.startswith("valid_") and \
           f != 'valid_lexical_complexity.txt' and \
           f != 'valid_readability.txt':
            file_names.append(join(path,f))

    matrix = []
    header = []
    matrix.append(header)
    
    for i, f in enumerate(file_names):
        feature_name = f[len(path) + len("valid_"):-4]

        # 判断delimiter
        delimiter = ''
        tmp = open(f, 'r')
        splits = len(next(tmp).split(' '))
        if splits > 1:
            delimiter = ' '
        else:
            delimiter = '\t'
        tmp.close()

        r = csv.reader(open(f, 'r'), delimiter = delimiter)

        if i == 0:
            header.append('essay_id')
            header.append(feature_name)
            for row in r:
                matrix.append(row)
        else:
            header.append(feature_name)
            for n, row in enumerate(r):
                matrix[n + 1].append(row[1])

    r = csv.reader(open(join(path, "valid_lexical_complexity.txt"), 'r'), delimiter = '\t')
    feature_names = next(r)[1:]
    header.extend(feature_names)
    for n, row in enumerate(r):
        matrix[n + 1].extend(row[1:-1]) # 最后一列多出一个制表符

    r = csv.reader(open(join(path, "valid_readability.txt"), 'r'), delimiter = '\t')
    for i in range(1, 6):
        header.append("readability" + str(i))
    for n, row in enumerate(r):
        matrix[n + 1].extend(row[1:])

    w = csv.writer(open('../data/features/all_valid.txt', 'w'), delimiter = '\t')
    w.writerows(matrix)

prep_train()
prep_valid()
