from corpus_reader import read_dataset
from word_extraction import remove_puntuations
from numpy.random import choice

from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble

import numpy as np
import re
from time import time

from task2_features import get_features, get_num_features


### SETTINGS ###

n_features = get_num_features()


### FUNCTIONS ###

def get_num_pairs(data):
    n = 0
    for doc in data:
        for sentence in doc.sentences:
            n += len(sentence.pairs)
    return n

def preallocate_feature_matrix(data, n_features):
    n_pairs = get_num_pairs(data)
    return np.zeros((n_pairs, n_features))

class_index = {
        'null': 0,
        'advise' : 1,
        'effect':2,
        'int':3,
        'mechanism':4
    }
class_mapping = {v: k for k, v in class_index.items()}
def create_y(data):
    y = []
    for doc in data:
        for sentence in doc.sentences:
            for pair in sentence.pairs:
                label = class_index["null"]
                if pair.ddi == "true":
                    label = class_index[pair.type]
                y.append(label)
    return y

### SPLIT DATASET ###

data = read_dataset()
n_docs = len(data)

np.random.seed(42)

train_amount = 0.7
train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
test_ids = [i for i in range(n_docs) if i not in train_ids]

training = [data[i] for i in train_ids]
test = [data[i] for i in test_ids]

print("%i training documents" % len(training))
print("%i test documents" % len(test))


### TRAINING ###

X = preallocate_feature_matrix(training, n_features)

i_pair = 0
for doc in training:
    for sentence in doc.sentences:
        for pair in sentence.pairs:
            X[i_pair, :] = get_features(pair, sentence)
            i_pair += 1

y = create_y(training)

print("training...")
start = time()
classifier = svm.SVC(probability = True)
classifier.fit(X, y)
print ("Done in " + str(time() - start) + "s")






### EVALUATION ###

correct = 0
total = 0

precision_count = {
    'null': {},
    'advise' : {},
    'effect': {},
    'int': {},
    'mechanism': {}
}

for k in precision_count:
    precision_count[k] = {
        "true_positive" : 0,
        "false_positive" : 0,
        "false_negative" : 0,
        "correct" : 0
    }
    

for tdoc in test:
    for sentence in tdoc.sentences:
        if len(sentence.entities) >= 2:
            for p in sentence.pairs:
                f = get_features(p, sentence).reshape(1,-1)
                
                pred = class_mapping[classifier.predict(f)[0]]
                true = p.getLabel()
                
                if pred == true:
                    precision_count[pred]["true_positive"] += 1
                elif pred != true:
                    precision_count[pred]["false_positive"] += 1
                    precision_count[true]["false_negative"] += 1
                
                total += 1
                
                if pred == true:
                    correct += 1

def print_score(count, label):
    print("> %i true positives" % count[label]["true_positive"])
    print("> %i false positives" % count[label]["false_positive"])
    print("> %i false negatives" % count[label]["false_negative"])
    precision = float(count[label]["true_positive"]) / (count[label]["true_positive"] + count[label]["false_positive"]) if (count[label]["true_positive"] + count[label]["false_positive"]) != 0 else 0 
    recall = float(count[label]["true_positive"]) / (count[label]["true_positive"] + count[label]["false_negative"]) if (count[label]["true_positive"] + count[label]["false_negative"]) != 0 else 0

    print("> recall: %f" % recall)
    print("> precision: %f" % precision)
    print("-"*5)

for k in precision_count:
    print("Precision for class "+k+":")
    print_score(precision_count, k)

accuracy = round(100*100*float(correct)/total)/100
print("correct: "+str(correct))
print("total: "+str(total))
print("accuracy: "+str(accuracy)+ "%")
