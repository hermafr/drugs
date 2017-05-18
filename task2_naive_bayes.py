from corpus_reader import read_dataset
from feature_reader import FeatureReader
from naive_bayes import NaiveBayes
from numpy.random import choice
from multiclass_ngram_naive_bayes2 import confusion_matrix, print_confusion_matrix
from k_fold import k_folds
import numpy as np
import re

TEST = True

def word_list(text):
    return [word for word in re.split("\W+", text) if len(word) > 0]

def contains_negation(word_list):
    return "no" in word_list or "not" in word_list or "none" in word_list or "never" in word_list

def get_feature_list(feature_reader, pair, sentence):
    """ creates featurs of a pair in a sentence
    """
    features = []
    
    # parsing features
    tag = fr.get_lca_tag(pair.id)
    features.append(("lca_tag", tag))
    word = fr.get_lca_word(pair.id)
    features.append(("lca_word", word))
    
    # words in between
    words_in_between = word_list(pair.textBetween)
    for word in words_in_between:
        features.append(("word_in_between", word))
    num_words_between = len(words_in_between)
    features.append(("num_words_between", num_words_between))
    negation_in_between = contains_negation(words_in_between)
    features.append(("negation_in_between", negation_in_between))
    
    # sentence features
    sentence_words = word_list(sentence)
    #for word in sentence_words:
    #    features.append(("word_in_sentence", word))
    #sentence_length = len(sentence_words)
    #features.append(("sentence_length", sentence_length))
    sentence_negation = contains_negation(sentence_words)
    features.append(("sentence_negation", sentence_negation))
    
    # key-words
    key_word_increase = "increase" in sentence_words or "increases" in sentence_words
    key_word_decrease = "decrease" in sentence_words or "decreases" in sentence_words
    features.append(("key_word_increase", key_word_increase))
    features.append(("key_word_decrease", key_word_decrease))
    
    key_word_no = "no" in sentence_words
    key_word_not = "not" in sentence_words
    key_word_none = "none" in sentence_words
    key_word_never = "never" in sentence_words
    features.append(("key_word_no", key_word_no))
    features.append(("key_word_not", key_word_not))
    features.append(("key_word_none", key_word_none))
    features.append(("key_word_never", key_word_never))
    key_words_no_effect = "no" in sentence_words and "effect" in sentence_words
    features.append(("key_words_no_effect", key_words_no_effect))
    
    # entity features
    #drug1_type = pair.e1.type
    #drug2_type = pair.e2.type
    #features.append(("drug1_type", drug1_type))
    #features.append(("drug2_type", drug2_type))
    #drug1_name = pair.e1.text
    #drug2_name = pair.e2.text
    #features.append(("drug_name", drug1_name))
    #features.append(("drug_name", drug2_name))
    
    same_drug = pair.e1.text.lower() == pair.e2.text.lower()
    features.append(("same_drug", same_drug))
    
    return features

if __name__ == "__main__":
    np.random.seed(42)
    
    data = read_dataset()
    n_docs = len(data)
    
    n_folds = 10
    folds = k_folds(n_docs, n_folds)
    if TEST:
        n_folds = 1
    
    classes = ["int", "effect", "none", "mechanism", "advise"]
    cv_results = {}
    cv_precisions = []
    cv_recalls = []
    cv_fs = []
    cv_2class_precisions = []
    cv_2class_recalls = []
    cv_2class_fs = []
    
    for i_f in range(n_folds):
        print("*" * 20)
        print("FOLD %i" % i_f)
        
        if not TEST:
            train_ids = folds[i_f][0]
            test_ids = folds[i_f][1]
            training = [data[i] for i in train_ids]
            test = [data[i] for i in test_ids]
        else:
            training = data
            test = read_dataset(test = True, task = 2)
        
        print("%i training documents" % len(training))
        print("%i test documents" % len(test))
        
        ## initialisation
        
        fr = FeatureReader()
        
        nb = NaiveBayes(classes)
        
        nb.register_feature("lca_tag")
        nb.register_feature("lca_word")
        nb.register_feature("word_in_between")
        nb.register_feature("num_words_between")
        nb.register_feature("negation_in_between")
        nb.register_feature("sentence_length")
        nb.register_feature("word_in_sentence")
        nb.register_feature("sentence_negation")
        nb.register_feature("drug1_type")
        nb.register_feature("drug2_type")
        nb.register_feature("drug_name")
        nb.register_feature("key_word_increase")
        nb.register_feature("key_word_decrease")
        nb.register_feature("key_word_no")
        nb.register_feature("key_word_not")
        nb.register_feature("key_word_none")
        nb.register_feature("key_word_never")
        nb.register_feature("key_word_no_effect")
        nb.register_feature("same_drug")
        nb.register_feature("key_words_no_effect")
        
        ## training
        
        for doc in training:
            for sentence in doc.sentences:
                for pair in sentence.pairs:
                    c = pair.type if pair.type != None else "none"
                    nb.count_class(c)
                    feature_list = get_feature_list(fr, pair, sentence.text)
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
                    features = get_feature_list(fr, pair, sentence.text)
                    prediction = nb.classify(features, use_prior=True)
                    truth = pair.type if pair.type != None else "none"
                    #print((pair.e1.text, pair.e2.text, tag, prediction, truth))
                    gt_pred_pairs.append((pair.id, truth, prediction))
        
        ## evaluation
        
        counters = {}
        for c in nb.classes:
            counters[c] = [0, 0, 0]  # true positives, false positive, false negatives
        
        tp = 0
        fp = 0
        fn = 0
        
        c2_tp = 0
        c2_fp = 0
        c2_fn = 0
        
        for pair in gt_pred_pairs:
            truth = pair[1]
            pred = pair[2]
            
            if pred == truth:  # correct prediction
                counters[pred][0] = counters[pred][0] + 1
                if pred != "none":
                    tp += 1
            elif pred == "none":  # false negative
                counters[pred][1] = counters[pred][1] + 1
                counters[truth][2] = counters[truth][2] + 1
                fn += 1
            else:  # wrong prediction
                counters[pred][1] = counters[pred][1] + 1  # false positive of predicted class
                counters[truth][2] = counters[truth][2] + 1  # false negative of true class
                fp += 1
                if truth != "none":  # if truth is none it's just a false positive
                    fn += 1
            
            if pred != "none" and truth != "none":
                c2_tp += 1
            elif pred != "none" and truth == "none":
                c2_fp += 1
            elif pred == "none" and truth != "none":
                c2_fn += 1
        
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
        
        micro_precision = tp / (tp + fp)
        micro_recall = tp / (tp + fn)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        cv_precisions.append(micro_precision)
        cv_recalls.append(micro_recall)
        cv_fs.append(micro_f1)
        
        c2_precision = c2_tp / (c2_tp + c2_fp)
        c2_recall = c2_tp / (c2_tp + c2_fn)
        c2_f1 = 2 * c2_precision * c2_recall / (c2_precision + c2_recall)
        cv_2class_fs.append(c2_f1)
        
        print("%s: %s, p=%f, r=%f, f=%f" % ("2class",
                                            str([c2_tp, c2_fp, c2_fn]),
                                            c2_precision,
                                            c2_recall,
                                            c2_f1
                                            ))
        
        print("%s: %s, p=%f, r=%f, f=%f" % ("micro",
                                            str([tp, fp, fn]),
                                            micro_precision,
                                            micro_recall,
                                            micro_f1
                                            ))
    
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
    print("avg. 2c f = %f" % np.mean(cv_2class_fs))
    print("avg. micro f = %f" % np.mean(cv_fs))
