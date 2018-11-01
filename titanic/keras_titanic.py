# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     keras_titanic
   Description :
   Author :       Happiness
   date：          2018/10/29
-------------------------------------------------
   Change Activity:
                   2018/10/29:
-------------------------------------------------
"""
__author__ = 'Happiness'

import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DIR_TRAIN = os.getcwd() + "\\data\\train.csv"
DIR_TEST = os.getcwd() + "\\data\\test.csv"

test_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
train_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
               'Embarked']


def model():
    '''
    定义模型
    :return: keras model
    '''
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    return model


def convertModel(keras_model):
    '''
    将keras的模型转为estimator
    :param keras_model: keras模型
    :return: estimator
    '''
    return keras.estimator.model_to_estimator(keras_model)


def loadData(is_train, dir):
    if is_train:
        names = train_names
    else:
        names = test_names
    data = pd.read_csv(dir, header=0, names=names)
    data.pop('Cabin')
    data.pop('Name')
    data.pop('Ticket')
    return data


def progressData(data):
    # 补充完整数据
    passengerId = data.pop('PassengerId')
    mean_age = round(data.mean()['Age'], 1)
    mean_fare = round(data.mean()['Fare'], 1)
    data = data.fillna({'Age': mean_age, 'Fare': mean_fare})
    data = data.fillna(method='ffill')
    return data, passengerId


def splitData(datas, labels, splite):
    return train_test_split(datas, labels, test_size=splite, random_state=42)


# def feature():
#     feature_columns = []
#     # Pclass 分类特征列
#     feature_columns.append(tf.feature_column.numeric_column(key='Pclass', dtype=tf.int32))
#     # sex 分类词汇列
#     feature_columns.append(tf.feature_column.indicator_column(
#         tf.feature_column.categorical_column_with_vocabulary_list(key='Sex', vocabulary_list=['male', 'female'])))
#     # Age 数值列
#     feature_columns.append(
#         tf.feature_column.numeric_column(key='Age', normalizer_fn=lambda x: (x - mean_age) / std_age))
#     # SibSp 数值列
#     feature_columns.append(tf.feature_column.numeric_column(key='SibSp'))
#     # Parch 数值列
#     feature_columns.append(tf.feature_column.numeric_column(key='Parch'))
#     # Ticket 经过哈希处理的列
#     # feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket(key='Ticket',hash_bucket_size=))
#     # Fare 数值列
#     feature_columns.append(
#         tf.feature_column.numeric_column(key='Fare', normalizer_fn=lambda x: (x - mean_fare) / std_fare))
#     # Embarked 分类词汇列
#     feature_columns.append(tf.feature_column.indicator_column(
#         tf.feature_column.categorical_column_with_vocabulary_list(key='Embarked', vocabulary_list=['C', 'S', 'Q'])))

def generator(data, lables, batch_size):
    idx = np.arange(len(data))
    print(len(data))
    np.random.shuffle(idx)
    print(data.columns)
    batchs = [idx[range(batch_size * i, min(len(data), batch_size * (i + 1)))] for i in
              range(int(len(data) / batch_size + 1))]
    while True:
        for i in batchs:
            xx = data.loc[i, :]
            yy = lables.loc[i]
            yield (xx, yy)


if __name__ == '__main__':
    kerasModel = model()
    # estimator = convertModel(kerasModel)
    data_train = loadData(is_train=True, dir=DIR_TRAIN)
    data_train, _ = progressData(data_train)
    y = data_train.pop('Survived')
    data_test = loadData(is_train=False, dir=DIR_TEST)
    data_test, passengerId = progressData(data_test)

    X_train, X_test, y_train, y_test = splitData(data_train, y, 0.2)

    a = generator(X_train, y_train, 10)
    # kerasModel.fit_generator()
    for i in range(2):
        xx,yy = a.__next__()
        print(xx)

    # input_fn_train = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=128, num_epochs=1000,
    #                                                      shuffle=True)
    #
    # estimator.train(input_fn=input_fn_train, steps=1000)
