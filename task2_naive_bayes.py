from corpus_reader import read_dataset
from feature_reader import FeatureReader
from naive_bayes import NaiveBayes
from numpy.random import choice
from multiclass_ngram_naive_bayes2 import confusion_matrix, print_confusion_matrix
from k_fold import k_folds
import numpy as np
import re

def get_feature_list(feature_reader, pair):
    features = []
    tag = fr.get_lca_tag(pair.id)
    features.append(("lca_tag", tag))
    word = fr.get_lca_word(pair.id)
    features.append(("lca_word", word))
    words_in_between = [word for word in re.split("\W+", pair.textBetween) if len(word) > 0]
    for word in words_in_between:
        features.append(("word_in_between", word))
    return features

if __name__ == "__main__":
    np.random.seed(42)
    
    data = read_dataset()
    n_docs = len(data)
    
    n_folds = 10
    folds = k_folds(n_docs, n_folds)
    
    classes = ["none", "brand", "drug", "drug_n", "group"]
    cv_results = {}
    
    for i_f in range(n_folds):
        print("*" * 20)
        print("FOLD %i" % i_f)
        
        train_ids = folds[i_f][0]
        test_ids = folds[i_f][1]
        training = [data[i] for i in train_ids]
        test = [data[i] for i in test_ids]
        
        print("%i training documents" % len(training))
        print("%i test documents" % len(test))
        
        ## initialisation
        
        fr = FeatureReader()
        
        classes = ["int", "effect", "none", "mechanism", "advise"]
        nb = NaiveBayes(classes)
        
        nb.register_feature("lca_tag")
        nb.register_feature("lca_word")
        nb.register_feature("word_in_between")
        
        ## training
        
        for doc in training:
            for sentence in doc.sentences:
                for pair in sentence.pairs:
                    c = pair.type if pair.type != None else "none"
                    nb.count_class(c)
                    feature_list = get_feature_list(fr, pair)
                    for feature in feature_list:
                        feature_name = feature[0]
                        feature_value = feature[1]
                        nb.count_feature(feature_name, feature_value, c)
        
        ## test
        
        gt_pred_pairs = []
        
        for doc in test:
            for sentence in doc.sentences:
                #print("")
                #print(sentence.text)
                for pair in sentence.pairs:
                    features = get_feature_list(fr, pair)
                    prediction = nb.classify(features, use_prior=True)
                    truth = pair.type if pair.type != None else "none"
                    #print((pair.e1.text, pair.e2.text, tag, prediction, truth))
                    gt_pred_pairs.append((pair.id, truth, prediction))
        
        ## evaluation
        
        counters = {}
        for c in nb.classes:
            counters[c] = [0, 0, 0]  # true positives, false positive, false negatives
        
        for pair in gt_pred_pairs:
                        truth = pair[1]
                        pred = pair[2]
                        if pred == truth:  # correct prediction
                            counters[pred][0] = counters[pred][0] + 1
                        elif pred == "none":  # false negative
                            counters[truth][2] = counters[truth][2] + 1
                        else:  # wrong prediction
                            counters[pred][1] = counters[pred][1] + 1  # false positive of predicted class
                            counters[truth][2] = counters[truth][2] + 1  # false negative of true class
        
        #conf_matrix = {}
        #confusion_matrix(conf_matrix, gt_pred_pairs)
        #print_confusion_matrix(conf_matrix)
        
        for c in sorted(counters):
                true_positives = counters[c][0]
                false_positives = counters[c][1]
                false_negatives = counters[c][2]
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                print("%s: %s, p=%f, r=%f, f=%f" % (c, str(counters[c]), precision, recall, f1))
                
                if c not in cv_results:
                    cv_results[c] = [[],[],[]]
                cv_results[c][0].append(precision)
                cv_results[c][1].append(recall)
                cv_results[c][2].append(f1)
    
    print("*" * 20)
    f_sum = 0
    for c in sorted(cv_results):
        f = np.mean(cv_results[c][2])
        f_sum += f
        print("%s: p=%f, r=%f, f=%f" % (c,
                                        np.mean(cv_results[c][0]),
                                        np.mean(cv_results[c][1]),
                                        f))
    print("*" * 20)
    print("avg. f = %f" % (f_sum / len(cv_results)))
