# -*- coding: utf-8 -*-
import time

import util
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

fn_desc = util.project_dir + os.path.sep + 'data' + os.path.sep + 'desc.txt'

def read_desc():
    f_desc = open(fn_desc, mode='r')
    lines = f_desc.readlines()
    attributes = []
    categorys_map = {}
    for line in lines:
        line = line.strip().rstrip('.')
        attribute, categorys = tuple(line.split(':'))
        attributes.append(attribute)
        categorys_map[attribute] = {category.strip(): index for index, category in enumerate(categorys.split(','))}
    return attributes,categorys_map

def preprocessing(attributes, categorys_map, fn_data):
    df = pd.read_csv(fn_data, header=None, names=attributes)
    print(df.shape)
    # replace '?' by NaN
    df = df.replace(r'\?', np.nan, regex=True)
    df.dropna(inplace=True)
    print(df.shape)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].map(lambda x: categorys_map[column][x])
            # print(df.info())
    return df

def train(df_train,df_test):
    X_train = df_train.drop('income',axis=1)
    y_train = df_train['income']
    X_test = df_train.drop('income', axis=1)
    y_test = df_train['income']

    start = time.time()
    lr = LogisticRegression(penalty='l1', tol=0.01)
    lr.fit(X_train,y_train)
    start = cost_times(start,'lr.fit')
    score = lr.score(X_test,y_test)
    start = cost_times(start, 'lr.score')
    print(score)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train)
    start = cost_times(start, 'rfc.fit')
    score = rfc.score(X_test, y_test)
    start = cost_times(start, 'rfc.score')
    print(score)

    from sklearn import svm
    svm = svm.SVC()
    svm.fit(X_train,y_train)
    start = cost_times(start, 'svc.fit')
    score = svm.score(X_test,y_test)
    start = cost_times(start, 'svc.score')
    print(score)



def cost_times(start,step):
    end = time.time()
    print(str((end - start)) + step)
    return end

if __name__ == '__main__':
    attributes, categorys_map = read_desc()
    fn_data = util.project_dir + os.path.sep + 'data' + os.path.sep + 'adult.data.csv'
    df_train = preprocessing(attributes, categorys_map, fn_data)
    fn_data = util.project_dir + os.path.sep + 'data' + os.path.sep + 'adult.test.csv'
    df_test = preprocessing(attributes, categorys_map, fn_data)
    train(df_train, df_test)
