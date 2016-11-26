# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import Imputer


def preprocess_column_age(data, attributes, category_map):
    # data is just read from file and did not delete or change any value
    df = data.replace(r'\?', np.nan, regex=True)
    column_data = df['age']
    if getNaNColumnNumbers(df['age']) > 0:
        imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
        imr = imp.fit(column_data)
        transformed_data = imp.transform(column_data.values)
        print column_data.values
        print '-------------------'
        print transformed_data
        df['age'] = transformed_data[0]

        count = getNaNColumnNumbers(df['age'])
    return


# for continuous values
def preprocess_column(data, attribute):
    column_data = data[attribute]
    if getNaNColumnNumbers(column_data, attribute) > 0:
        imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
        imr = imp.fit(column_data)
        transformed_data = imp.transform(column_data.values)
        print column_data.values
        print '-------------------'
        print transformed_data
        data[attribute] = transformed_data[0]
        getNaNColumnNumbers(data[attribute], attribute)
    return


import re
import math


def getNaNColumnNumbers(data, column_name='default'):
    row_index = []
    for row in data:
        # if type(row) != float and math.isnan(row) or (type(row) == 'str' and re.match(r'\?', row)):
        if math.isnan(row):
            row_index.append(data.index(row))
    print 'NaN Count ', column_name, ': ', len(row_index)
    return len(row_index)


def getNaNFromColumnWithAllowedValues(data, allowed_values, column_name='default'):
    row_index_list = []
    for row_index in range(len(data)):
        # if type(row) != float and math.isnan(row) or (type(row) == 'str' and re.match(r'\?', row)):
        row = data[row_index]
        if row in allowed_values:
            continue
        else:
            row_index_list.append(row_index)
    print 'NaN Count ', column_name, ': ', len(row_index_list)
    return len(row_index_list)
    return


def predict_value(data, predicting_column_name, using_other_column_list, category_map):
    x_train = []
    y_train = []

    x_test = []
    x_test_index = []

    for i in range(len(data)):
        x_train_row = []
        x_test_row = []

        if data[predicting_column_name][i] in category_map[predicting_column_name]:  # this is a missing row
            for column_name in using_other_column_list:
                x_train_row.append(data[column_name][i])
            x_train.append(x_train_row)
            y_train.append(data[predicting_column_name][i])
        else:
            for column_name in using_other_column_list:
                x_test_row.append(data[column_name][i])
            x_test.append(x_test_row)
            x_test_index.append(i)
    # print len(x_train), len(x_train[0])
    # print len(y_train), len(y_train[0])
    # print len(x_test), len(x_test[0])
    import pandas as pd
    df = pd.DataFrame(x_train, columns=using_other_column_list)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].map(lambda x: category_map[column][x])
            # print(df.info())
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(df, y_train)

    test_df = pd.DataFrame(x_test, columns=using_other_column_list)
    for column in df.columns:
        if test_df[column].dtype == 'object':
            test_df[column] = test_df[column].map(lambda x: category_map[column][x])
    predict_result = rfc.predict(test_df)
    print("predict result : ", predict_result)

    for i in range(len(x_test_index)):
        row_index = x_test_index[i]
        data[predicting_column_name][row_index] = predict_result[i]

    getNaNFromColumnWithAllowedValues(data[predicting_column_name], category_map[predicting_column_name], predicting_column_name)
    return data

# attributes, categorys_map = main.read_desc()
# fn_data = util.project_dir + os.path.sep + 'data' + os.path.sep + 'adult.data.csv'
# column_age('../data/adult.data.csv')
# df_train = main.preprocessing(attributes, categorys_map, fn_data)
# fn_data = util.project_dir + os.path.sep + 'data' + os.path.sep + 'adult.test.csv'
# df_test = main.preprocessing(attributes, categorys_map, fn_data)
