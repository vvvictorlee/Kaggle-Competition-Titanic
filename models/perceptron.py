# coding: utf-8

import pandas as pd
import numpy as np
from collections import Counter


# train_data = pd.read_csv('train_processed.csv', dtype=str, usecols=range(1,10))
# actual = pd.read_csv('gender_submission.csv', usecols=[1], dtype=str)
# test_data = pd.read_csv('test_processed.csv', usecols=range(1,9), dtype=str)
# train_data.head()
# train_X = train_data.drop(['Survived'], axis=1)
# # train_X.head()
# train_Y = train_data['Survived']
# # train_Y.head()
# test_X = test_data
# # test_X.head()
# test_Y = actual


# one-hot encoding
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# enc_train_X = enc.fit_transform(train_X)
# enc_test_X = enc.fit_transform(test_X)
# enc_train_X = pd.DataFrame(enc_train_X.toarray())
# enc_test_X = pd.DataFrame(enc_test_X.toarray())
# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler(feature_range=(-1,1))
# enc_train_X = min_max_scaler.fit_transform(enc_train_X)
# enc_test_X = min_max_scaler.fit_transform(enc_test_X)
# enc_train_X = pd.DataFrame(enc_train_X)
# enc_test_X = pd.DataFrame(enc_test_X)


# make prediction for each row
def predict(row, weights, bias):
    activation = bias
    activation += sum(row*weights)
    return 1.0 if activation >= 0.0 else 0.0


# In[44]:

def train(train_data, n_epoch, learning_rate):
    weights = np.random.uniform(-1,1,train_data.shape[1]-1)
    bias = 0
    for epoch in range(n_epoch):
        error_sum = 0.0
        for row in train_data.values.astype('float'):
            pred = predict(row[:-1], weights, bias)
            error = row[-1]-pred
            error_sum += error**2
            bias += learning_rate*error
            weights += learning_rate*error*row[:-1]
#         print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, error_sum))
    return weights, bias


def perceptron(train_data, test_X, test_Y, learning_rate, n_epoch):
    predictions = list()
    weights, bias = train(train_data, n_epoch, learning_rate)
    for row in test_X.values.astype('float'):
        prediction = predict(row, weights, bias)
        predictions.append(prediction)
    return predictions


def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual))*100.0


def evaluate(train_data, test_X, test_Y, learning_rate, n_epoch):
    predicted = perceptron(train_data, test_X, test_Y, learning_rate, n_epoch)
    acc = accuracy(test_Y, predicted)
#     print(predicted)
    return acc



# best_score =0
# for i in range(10):
#     train_data = pd.concat([enc_train_X, train_Y], axis=1)
#     # print(train_data.head())
#     test_X = enc_test_X
#     actual = test_Y.values.astype('float')
#     learning_rate = 0.1
#     n_epoch = 100
#     score = evaluate(train_data, test_X, actual, learning_rate, n_epoch)
#     if score > best_score:
#         best_score = score
# print('The accuracy for SGD is %f' % best_score)




