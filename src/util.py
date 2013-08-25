# coding=utf-8

import os
from os.path import join
import csv

def get_essay_set_range(category):
    path = ""
    if category == "train":
        path = "../data/training_set.tsv"
    elif category == "valid":
        path = "../data/valid_set.tsv"

    r = csv.reader(open(path, 'r'), delimiter = '\t')
    next(r) # 去掉header

    set_range = {} # 记录每个essay set的开始和结束
    first_row = next(r)
    essay_id = first_row[0]
    essay_set = first_row[1]
    set_range[essay_set] = [essay_id]

    previous_set = essay_set
    previous_id = essay_id
    for row in r:
        essay_id = row[0]
        essay_set = row[1]

        if essay_set != previous_set:
            set_range[previous_set].append(previous_id)
            set_range[essay_set] = [essay_id]
        
        previous_id = essay_id
        previous_set = essay_set

    set_range[previous_set].append(previous_id)

    return set_range

# 判断是否有header
def has_header(f):
    delimiter = get_delimiter(f)
    r = csv.reader(open(f, 'r'), delimiter = delimiter)
    s = next(r)[0]
    try:
        float(s)
        return False
    except ValueError:
        return True

# 判断delimiter
def get_delimiter(f):
    delimiter = ''
    tmp = open(f, 'r')
    splits = len(next(tmp).split(' '))
    if splits > 1:
        delimiter = ' '
    else:
        delimiter = '\t'
    tmp.close()
    return delimiter

def prep_features():
    prefixes = ("train_", "valid_")
    path = "../data/features/"

    suffixes = []
    prefix = prefixes[0]
    for f in os.listdir(path):
        suffix = f[len(prefix):]
        if f.startswith(prefix) and \
           suffix not in ("lexical_complexity.txt", 
                          "readability.txt", 
                          "ttr_phrase.tsv"):
            suffixes.append(suffix)

    for prefix in prefixes:
        file_names = []
        for suffix in suffixes:
            file_names.append(join(path, prefix + suffix))

        matrix = []
        header = []
        matrix.append(header)
        
        for i, f in enumerate(file_names):
            delimiter = get_delimiter(f)
            r = csv.reader(open(f, 'r'), delimiter = delimiter)

            feature_names = []
            if has_header(f):
                feature_names = next(r)[1:]
            else: # 从文件名中得到特征名
                feature_names.append(f[len(path) + len(prefix):-4])

            if i == 0:
                header.append('essay_id')
                header.extend(feature_names)
                for row in r:
                    # 添加新的一行
                    matrix.append(row)
            else:
                header.extend(feature_names)
                for n, row in enumerate(r):
                    matrix[n + 1].extend(row[1:])

        # 单独处理xxx_lexical_complexity.txt
        r = csv.reader(open(join(path, prefix + "lexical_complexity.txt"), 'r'), delimiter = '\t')
        feature_names = next(r)[1:]
        header.extend(feature_names)
        for n, row in enumerate(r):
            matrix[n + 1].extend(row[1:-1]) # 最后一列多出一个制表符

        # 单独处理xxx_ttr_phrase.tsv
        r = csv.reader(open(join(path, prefix + "ttr_phrase.tsv"), 'r'), delimiter = '\t')
        feature_names = next(r)[-3:] # 只取后三个
        header.extend(feature_names)
        for n, row in enumerate(r):
            matrix[n + 1].extend(row[-3:])

        # 单独处理xxx_readability.txt
        r = csv.reader(open(join(path, prefix + "readability.txt"), 'r'), delimiter = '\t')
        for i in range(1, 6):
            header.append("readability" + str(i))
        for n, row in enumerate(r):
            matrix[n + 1].extend(row[1:])

        target_file = join(path, "all_" + prefix[:-1] + ".txt")
        w = csv.writer(open(target_file, 'w'), delimiter = '\t')
        w.writerows(matrix)

