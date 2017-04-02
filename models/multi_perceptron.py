from collections import Counter
import math
import sys
import random
from random import seed
from random import randrange
import numpy as np 


# n_folds = 20
# l_rate = 0.01
# classes = [0,1,2,3,4]
# INPUT_FILE = sys.argv[1]
n_epoch = 20


class MultiClassPerceptron():

    # weights: list of dict<token, int>
    def __init__(self, classes, vectors_tfidf, n_epoch= n_epoch):
        self.classes = classes
        self.vectors_tfidf = vectors_tfidf
        self.n_epoch= n_epoch
        weight = {}
        for label, vector in self.vectors_tfidf:
            for token in vector:
                weight[token] = 0
        # the index corresponds to 0,1,2,3,4, the five categories
        self.weights = [dict(weight) for c in self.classes]
        self.bias = [0.5 for c in self.classes]
        # print self.weights


    # vectors_tfidf<list> of tuples (label<sting>, tfidf<dict<token: float>>)
    #  n_folds: int
    # return: list of folds: which is <list> of tuples (label<sting>, tfidf<dict<token: float>>)
    def cross_validation_split(self):
        dataset_split = list()
        dataset_copy = list(self.vectors_tfidf)
        fold_size = len(self.vectors_tfidf)/n_folds
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split



    # row: aka a dict
    # weights: a list of floats 
    #  return: 1.0 or 0.0
    def predict(self, row):

        # Initialize arg_max value, predicted class.
        argmax, predicted = 0, self.classes[0]

        # Multi-Class Decision Rule:
        for c in self.classes:
            # current_activation = self.bias[c]
            current_activation = 0
            for token in row:
                current_activation += row[token]*self.weights[c][token]
            # print current_activation
            if current_activation >= argmax:
                argmax, predicted = current_activation, c
        return predicted



    def f1_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        precision = correct/float(len(predicted)) 
        recall = correct/float(len(actual)) 
        return (2*precision*recall)/(precision+recall) * 100


    # train: train_set: list of tuples (label<sting>, tfidf<dict<token: float>>)
    # test: test_set: list of tuples (label<sting>, tfidf<dict<token: float>>)
    # return: list of int
    def perceptron(self, train, test):
        predictions = []
        self.train(train)
        for tup in test:
            row = tup[1]
            predicted = self.predict(row)
            predictions.append(predicted)
        return predictions



    def train(self, train):
        for epoch in range(self.n_epoch):
            sum_error = 0.0
            for tup in train:
                row = tup[1]
                predicted = self.predict(row)

                # Update Rule:
                label = tup[0]
                error = float(label - predicted)
                if not (label == predicted):
                    for token in row:
                        self.weights[predicted][token] -= row[token]*l_rate
                        # self.bias[predicted] -= l_rate*error
                        self.weights[label][token] += row[token]*l_rate
                        # self.bias[label] += l_rate*error
                sum_error += error**2

            print('>epoch=%s, weights=%.3f' % (epoch, sum_error))




    # perceptron
    # n_folds: int
    def evaluate_algorithm(self):
        folds = self.cross_validation_split()
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for tup in fold:
                tup_copy = tuple(tup)
                test_set.append(tup_copy)
            predicted = self.perceptron(train_set, test_set)
            actual = [row[0] for row in fold]
            f1 = self.f1_metric(actual, predicted)
            scores.append(f1)
        mean = sum(scores)/float(len(scores))
        return mean



# INPUT_FILE: csv
# return: vectors_tfidf<list> of tuples (label<sting>, tfidf<dict<token: float>>)
# def generate_tfidf(input):
#     # bag-of-words
#     vectors_bow = list()
#     dfs = Counter()

#     # words are not lemmatized
#     for line in input:
#         t = line.split('\t')
#         label = t[0]
#         document = t[1]
#         vector = (int(label), Counter(document.split()))
#         vectors_bow.append(vector)
#         dfs.update(vector[1].keys())

#     # document frequency
#     D = float(len(vectors_bow))

#     for token in dfs:
#         dfs[token] = 1.0 / math.log(D/dfs[token])

#     # tf-idf
#     vectors_tfidf = list()

#     for (label, bow) in vectors_bow:
#         tfidf = {token : dfs[token]*count for (token, count) in bow.items()}
#         vectors_tfidf.append((label, tfidf))
#     return vectors_tfidf



# if __name__ == "__main__":
#     input = open(INPUT_FILE)
#     vectors_tfidf = generate_tfidf(input)
#     # for c,each in vectors_tfidf:
#     #     print len(each)
#     sentiment_classifier = MultiClassPerceptron(classes, vectors_tfidf, n_epoch)
#     f1_score = sentiment_classifier.evaluate_algorithm()
#     print f1_score
    








