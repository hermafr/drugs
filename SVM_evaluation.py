import numpy as np

#import SVM_class
from SVM_class import SVM
from sklearn import svm
from DDIClassifier import DDIClassifier
from MostFrequentBetweenStrategy import MostFrequentBetweenStrategy

from corpus_reader import read_dataset
from numpy.random import choice

from time import time


print("Loading data")
data = read_dataset()
n_docs = len(data)

np.random.seed(42)

classes = ['null', 'advise', 'effect', 'int', 'mechanism']

train_amount = 0.7
train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
test_ids = [i for i in range(n_docs) if i not in train_ids]

training = [data[i] for i in train_ids]
test = [data[i] for i in test_ids]

print("%i training documents" % len(training))
print("%i test documents" % len(test))



ddi_clf = DDIClassifier(MostFrequentBetweenStrategy(nb_feature = 20),
    clf = svm.LinearSVC())

print("Creating feature matrix...")
trainingFeature = ddi_clf.getFeatureMatrix(training)
print("Done\n")

print("Creating labels...")
labels = ddi_clf.getLabels(training)
print("Done\n")

print("Fitting the model...")
ddi_clf.fit(trainingFeature, labels, verbose=True)
print("Done\n")


#### EVALUATE ####

correct = 0
total = 0

precision_count = { c : {} for c in classes}
    

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
                pred = ddi_clf.predictFromTextBetween(p.textBetween)
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
    tp = count[label]["true_positive"]
    fp = count[label]["false_positive"]
    fn = count[label]["false_negative"]
    
    print("> %i true positives" % tp)
    print("> %i false positives" % fp)
    print("> %i false negatives" % fn)
    precision = float(tp) / (tp + fp) if tp+fp != 0 else 0
    recall = float(tp) / (tp + fn) if tp+fn != 0 else 0

    print("> recall: %f" % recall)
    print("> precision: %f" % precision)
    print("-"*5)
    


for k in precision_count:
    print("Precision for class "+k+":")
    print_score(precision_count, k)
    
