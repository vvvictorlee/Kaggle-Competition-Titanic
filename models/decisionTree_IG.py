# coding: utf-8
import sys

# rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from training data
def attribute_count(rows):
    from collections import Counter
    counts = Counter([row[0] for row in rows])
    return counts

# rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from training data
def entropy(rows):
    counts = attribute_count(rows)
    entropy = 0
    from math import log
    log2 = lambda x: log(x)/log(2)
    for unit in counts.keys():
#       remember to use float
        p = float(counts[unit])/len(rows)
        entropy -= (p)*log2(p)
    return entropy


# rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from training data
# column: the column number indicating the attribute we are dealing with
def divideset(rows,column,value):
   split_function=lambda row:row[column]==value
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)


# the decisionNode data structure
class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

# rows: list of list of instance [classValue, attribute1Value, attribute2Value, ......] from training data
# func: assign the function to calculate the entropy level
def buildtree(rows,func=entropy):
    if len(rows)==0:
        return decisionnode()
    info_D = func(rows)
    best_gain=0.0
    best_pair=None
    best_sets=None
    from math import log
    log2 = lambda x: log(x)/log(2)
    for col in range(1, len(rows[0])):
        values = set(row[col] for row in rows)
        for value in values:
            (set1, set2) = divideset(rows, col, value)
            p = float(len(set1))/len(rows)
            # if p == 1, len(rows)==1, so GR=0 still
            if p == 0 or p == 1:
                continue
            infoa_D = p*(func(set1))+(1-p)*(func(set2))
            gain = info_D-infoa_D
            if gain > best_gain and len(set1)> 0 and len(set2) > 0:
                best_gain = gain
                best_pair = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_pair[0], value=best_pair[1], tb = trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=attribute_count(rows))


# observation: list of attributes
# tree: the root of subtree
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    v = observation[tree.col]
    branch = None
    if v==tree.value:
        branch = tree.tb
    else:
        branch = tree.fb
    return classify(observation, branch)


# This func is not used for the task
def prune(tree, mingain):
    if tree.tb and tree.fb:
        if tree.tb.results == None:
            prune(tree.tb, mingain)
        if tree.fb.results == None:
            prune(tree.fb, mingain)
    #   if both brances are leaves, then merge
        if tree.tb.results != None and tree.fb.results != None:
            tb, fb = [], []
            for k, v in tree.tb.results.items():
                tb += [[k]]*v
            for k, v in tree.fb.results.items():
                fb += [[k]]*v
        #   test the merge set VS separate two sets
            delta = entropy(tb+fb)-(entropy(tb)+entropy(fb))/2
            # print delta
            if delta < mingain:
                tree.fb, tree.tb = None, None
                tree.results = attribute_count(tb+fb)

# actual: list of actual classValues (corresponding to the number of the instance)
# predictions: list of predicted classValues (corresponding to the number of the instance)
def evaluate_accuracy(actual, predictions):
    count = 0
    for i, each in enumerate(actual):
        if each == predictions[i]:
            count += 1
    accuracy = float(count)/len(actual)
    return accuracy

# train_set = open(sys.argv[1])
# train_set = train_set.read()
# test_set = open(sys.argv[2])
# test_set = test_set.read()
# outputfile = sys.argv[3]

# train_set = train_set.split('\n')
# test_set = test_set.split('\n')
# rows = []
# for each in train_set:
#     if each != '':
#         each = each.split('\t')
#         rows.append(each)
# actual = []
# test_rows = []
# for each in test_set:
#     if each != '':
#         each = each.split('\t')
#         actual.append(each[0])
#         test_rows.append(each)

# root = buildtree(rows)
# # prune(root, 0.1)
# predictions = []
# for row in test_rows:
#     predicted = classify(row, root).most_common(1)[0][0]
#     predictions.append(predicted)

# # output file
# with open(outputfile, 'w') as out:
# 	out.write('Actual\tPredicted\n')
# 	for i, each in enumerate(predictions):
# 		out.write('%-6s\t%-9s\n' % (str(each), str(actual[i])))
# 	out.write('The accuracy of the program is ')
# 	out.write(str(evaluate_accuracy(actual, predictions)))




