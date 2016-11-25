# -*- coding: utf-8 -*-
import util
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

fn_data = util.project_dir + os.path.sep + 'data' + os.path.sep + 'adult.data.csv'
fn_desc = util.project_dir + os.path.sep + 'data' + os.path.sep + 'desc.txt'

f_desc = open(fn_desc,mode='r')
lines = f_desc.readlines()
attributes = []
categorys_map = {}
for line in lines:
    line = line.strip().rstrip('.')
    attribute, categorys = tuple(line.split(':'))
    attributes.append(attribute)
    categorys_map[attribute] = [category.strip() for category in categorys.split(',')]

data = pd.read_csv(fn_data,header=None,names=attributes)
print(data.info())
# replace '?' by NaN
data = data.replace(r'\?', np.nan, regex=True)
data.dropna(inplace=True)
print(data.shape)
# print(data.describe)
Y1 = []
Y2 = []
for edu_value in categorys_map['education']:
    a_edu_data = data[data['education'] == edu_value]
    tmp = a_edu_data[a_edu_data['income'] == '<=50K']
    Y1.append(len(tmp))
    tmp = a_edu_data[a_edu_data['income'] == '>50K']
    Y2.append(len(tmp)*(-1))
    for income in categorys_map['income']:
        tmp = a_edu_data[a_edu_data['income'] == income]
        print(edu_value + "\t" + income + "\t : " + str(len(tmp)))

n = len(categorys_map['education'])
X = range(1,n+1)

plt.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, Y2, facecolor='#ff9999', edgecolor='white')
for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.0f' % y, ha='center', va= 'bottom')
for x,y in zip(X,Y2):
    plt.text(x+0.4, y-1000, '%.0f' % y, ha='center', va= 'bottom')
plt.ylim(-6000,+8500)
plt.show()



