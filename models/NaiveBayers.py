import sys
from collections import defaultdict
# coding: utf-8

# classes<classValue, counts>
# attributes list of dicts<classValue+attributeValue, counts>
# rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from training data
def prepare(classes, attributes, rows):
    for row in rows: 
        classes[row[0]] += 1
        for i in range(1, len(row)):
#           key of a dict in attributes: class + attribute
            attributes[i][row[0]+row[i]] += 1  
            
# actual: list of actual classValues (corresponding to the number of the instance)
# predictions: list of predicted classValues (corresponding to the number of the instance)
def evaluate_accuracy(actual, predictions):
    # calculate the precision
    count = 0
    for i, each in enumerate(actual):
        if each == predictions[i]:
            count += 1
    accuracy = float(count)/len(actual)
    return accuracy

# classes<classValue, counts>
# attributes list of dicts<classValue+attributeValue, counts>
# test_rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from test data
# actual: list of actual classValues (corresponding to the number of the instance)
def test(classes, attributes, test_rows, actual):
    predictions = []
    for row in test_rows:
        best_p = 0.0
        best_class = None
        for c in classes.keys():
            p = float(classes[c])/sum(classes.values())
            for i in range(1, len(row)):
                p *= float(attributes[i][c+row[i]]+1)/(classes[c] + len(attributes[i]))
            if p >= best_p:
                best_p = p
                best_class = c
        predictions.append(best_class)
    return predictions





