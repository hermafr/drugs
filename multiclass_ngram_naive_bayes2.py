from naive_bayes import NaiveBayes
from pos_tagging import PosTagger, TaggedWord
from feature_computation import n_grams, starts_with_uppercase
from corpus_reader import read_dataset
from numpy.random import choice
from k_fold import k_folds
from baseline_drug_span_predictor import BaselineDrugSpanPredictor
import numpy as np

TEST = True

VERBOSE = False

class MulticlassPosNgramNaiveBayes:
    def __init__(self):
        self.classes = ["drug", "group", "brand", "drug_n", "none"]
        self.nb = NaiveBayes(self.classes)
        self.tagger = PosTagger()
        self.set_feature_names()
    
    def set_feature_names(self):
        self.ngram_feature_name = "ngram"
        self.nb.register_feature(self.ngram_feature_name)
        self.word_length_feature_name = "word_length"
        self.nb.register_feature(self.word_length_feature_name)
        self.first_letter_uppercase_feature_name = "first_letter_uppercase"
        self.nb.register_feature(self.first_letter_uppercase_feature_name)
        self.last_letter_s_feature_name = "last_letter_s"
        self.nb.register_feature(self.last_letter_s_feature_name)
        # lemmas:
        self.lemma_before1_feature_name = "lemma-1"
        self.lemma_before2_feature_name = "lemma-2"
        self.lemma_feature_name = "lemma"
        self.lemma_after1_feature_name = "lemma+1"
        self.lemma_after2_feature_name = "lemma+2"
        self.nb.register_feature(self.lemma_before1_feature_name)
        self.nb.register_feature(self.lemma_before2_feature_name)
        self.nb.register_feature(self.lemma_feature_name)
        self.nb.register_feature(self.lemma_after1_feature_name)
        self.nb.register_feature(self.lemma_after2_feature_name)
        # pos tags:
        self.pos_before1_feature_name = "pos-1"
        self.pos_before2_feature_name = "pos-2"
        self.pos_feature_name = "pos"
        self.pos_after1_feature_name = "pos+1"
        self.pos_after2_feature_name = "pos+2"
        self.nb.register_feature(self.pos_before1_feature_name)
        self.nb.register_feature(self.pos_before2_feature_name)
        self.nb.register_feature(self.pos_feature_name)
        self.nb.register_feature(self.pos_after1_feature_name)
        self.nb.register_feature(self.pos_after2_feature_name)
    
    def train(self, data):
        for doc in data:
            for sentence in doc.sentences:
                tagged_words = self.tagger.pos_tag(sentence.text)
                labels = self.get_ground_truth(tagged_words, sentence.entities)
                feature_list = self.tagged_words_to_features(tagged_words)
                for i in range(len(tagged_words)):
                    label = labels[i]
                    self.nb.count_class(label)
                    for f in feature_list[i]:
                        self.nb.count_feature(f[0], f[1], label)
    
    def tagged_words_to_features(self, tagged_words):
        feature_list = []
        for i in range(len(tagged_words)):
            tagged_word = tagged_words[i]
            features = []
            # n-grams:
            ngrams = n_grams(tagged_word.word)
            for ngram in ngrams:
                f = (self.ngram_feature_name, ngram)
                features.append(f)
            # word length:
            features.append((self.word_length_feature_name, len(tagged_word.word)))
            # uppercase:
            features.append((self.first_letter_uppercase_feature_name, starts_with_uppercase(tagged_word.word)))
            # last letter s:
            features.append((self.last_letter_s_feature_name, tagged_word.word[-1] == 's'))
            #lemmas:
            features.append((self.lemma_before2_feature_name,
                             tagged_words[i - 2].lemma if (i - 2) > 0 else "#"))
            features.append((self.lemma_before1_feature_name,
                             tagged_words[i - 1].lemma if (i - 1) > 0 else "#"))
            features.append((self.lemma_feature_name, tagged_word.lemma))
            features.append((self.lemma_after1_feature_name,
                             tagged_words[i + 1].lemma if (i + 1) < len(tagged_words) else "#"))
            features.append((self.lemma_after2_feature_name,
                             tagged_words[i + 2].lemma if (i + 2) < len(tagged_words) else "#"))
            # POS tags:
            features.append((self.pos_before2_feature_name,
                             tagged_words[i - 2].pos if (i - 2) > 0 else "#"))
            features.append((self.pos_before1_feature_name,
                             tagged_words[i - 1].pos if (i - 1) > 0 else "#"))
            features.append((self.pos_feature_name, tagged_word.pos))
            features.append((self.pos_after1_feature_name,
                             tagged_words[i + 1].pos if (i + 1) < len(tagged_words) else "#"))
            features.append((self.pos_after2_feature_name,
                             tagged_words[i + 2].pos if (i + 2) < len(tagged_words) else "#"))
            # add to list:
            feature_list.append(features)
        return feature_list
    
    def intervals_intersect(self, i1, i2):
        intersect = i1[0] == i2[0] and i1[1] == i2[1]
        return intersect
    
    def get_ground_truth(self, tagged_words, entities):
        labels = []
        for tw in tagged_words:
            label = "none"
            for e in entities:
                for o in e.char_offset:
                    if self.intervals_intersect(tw.span, o):
                        label = e.type
                #if [tw.span] == e.char_offset:
                #    label = e.type
            labels.append(label)
        return labels
    
    def classify_tagged_words(self, tagged_words):
        labels = []
        feature_list = self.tagged_words_to_features(tagged_words)
        for i in range(len(tagged_words)):
            label = self.nb.classify(feature_list[i])
            labels.append(label)
        return labels
    
    def classify_tagged_words_as_spans(self, tagged_words):
        labels = self.classify_tagged_words(tagged_words)
        return wordlist_labels_to_span_predictions(tagged_words, labels)
    
    def classify_spans(self, sentence):
        tagged_words = self.tagger.pos_tag(sentence)
        return self.classify_tagged_words_as_spans(tagged_words)

def get_spans_from_entities(entities):
    spans = []
    for e in entities:
        spans.append(e.char_offset)
    return spans

def get_texts_from_entities(entities):
    texts = []
    for e in entities:
        texts.append(e.text)
    return texts

def simple_concatenation(words, spans):
    new_words = []
    new_spans = []
    for i in range(len(words)):
        if len(new_spans) == 0:
            new_words.append(words[i])
            new_spans.append(spans[i])
        else:
            if spans[i][0][0] == new_spans[-1][0][1] + 1:
                new_words[-1] = new_words[-1] + " " + words[i]
                new_spans[-1][0] = (new_spans[-1][0][0], spans[i][0][1])
            else:
                new_words.append(words[i])
                new_spans.append(spans[i])
    return new_words, new_spans

def get_gt_prediction_pairs(tagged_words, entities, labels):
    pairs = []
    n_words = len(tagged_words)
    n_entities = len(entities)
    entities_classified_as = ["none"] * n_entities
    # get ground truth of predicted words
    for i in range(n_words):
        if labels[i] != "none":
            entity_found = False
            span = [tagged_words[i].span]
            for j in range(n_entities):
                if span == entities[j].char_offset:
                    pairs.append((tagged_words[i].word, entities[j].type, labels[i]))
                    entities_classified_as[j] = labels[i]
                    entity_found = True
                    break
            if not entity_found:
                pairs.append((tagged_words[i].word, "none", labels[i]))
    # get missing predictions
    for i in range(n_entities):
        if entities_classified_as[i] == "none":
            pairs.append((entities[i].text, entities[i].type, entities_classified_as[i]))
    return pairs

def confusion_matrix(m, pairs):
    for p in pairs:
        truth = p[1]
        pred = p[2]
        if truth not in m:
            m[truth] = {}
        if pred not in m[truth]:
            m[truth][pred] = 0
        m[truth][pred] = m[truth][pred] + 1

def print_confusion_matrix(m):
    for truth in sorted(m):
        for pred in sorted(m[truth]):
            print(truth, pred, m[truth][pred])

def wordlist_labels_to_span_predictions(tagged_words, labels):
    span_preds = []
    for i in range(len(tagged_words)):
        if labels[i] != "none":
            span_preds.append((tagged_words[i].word, [tagged_words[i].span], labels[i]))
    return span_preds

def get_gt_prediction_pairs_from_spans(span_labels, entities):
    entity_labels = ["none"] * len(entities)
    pairs = []
    # try to match labels with entities
    for span_label in span_labels:
        entity_found = False
        for e_i, entity in enumerate(entities):
            if span_label[1] == entity.char_offset:
                pairs.append((span_label[0],  # the word
                              entity.type,  # truth
                              span_label[2]  # prediction
                              ))
                entity_labels[e_i] = span_label[2]
                entity_found = True
                break
        if not entity_found:  # false prediction
            pairs.append((span_label[0],  # the word
                          "none",  # truth
                          span_label[2]  # prediction
                          ))
    # entities that are not predicted
    for e_i, entity in enumerate(entities):
        if entity_labels[e_i] == "none":
            pairs.append((entity.text,  # the word
                          entity.type,  # truth
                          "none"  # prediction
                          ))
    return pairs

if __name__ == "__main__":
    np.random.seed(42)    
    
    data = read_dataset()
    n_docs = len(data)
    
    n_folds = 10
    folds = k_folds(n_docs, n_folds)
    
    classes = ["none", "brand", "drug", "drug_n", "group"]
    cv_results = {}
    cv_precisions = []
    cv_recalls = []
    cv_fs = []
    cv_2class_precisions = []
    cv_2class_recalls = []
    cv_2class_fs = []
    
    if TEST:
        n_folds = 1
    
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
            test = read_dataset(test = True, task = 1)
        
        print("%i training documents" % len(training))
        print("%i test documents" % len(test))
        
        # TODO nb = MulticlassPosNgramNaiveBayes()
        nb = BaselineDrugSpanPredictor()
        nb.train(training)
        
        ## test
        
        gt_pred_pairs = []
        
        for doc in test:
            for sentence in doc.sentences:
                span_labels = nb.classify_spans(sentence.text)
                new_gt_pred_pairs = get_gt_prediction_pairs_from_spans(span_labels, sentence.entities)
                for new_pair in new_gt_pred_pairs:
                    gt_pred_pairs.append(new_pair)
        
        ## evaluation
        
        conf_matrix = {}
        counters = {}
        for c in classes:
            counters[c] = [0, 0, 0]  # true positives, false positive, false negatives
        
        tp = 0
        fp = 0
        fn = 0
        
        c2_tp = 0
        c2_fp = 0
        c2_fn = 0
        
        fn_gaps = 0
        
        for pair in gt_pred_pairs:
            truth = pair[1]
            pred = pair[2]
            
            if pred == truth:  # correct prediction
                counters[pred][0] = counters[pred][0] + 1
                if pred != "none":
                    tp += 1
            elif pred == "none":  # false negative
                counters[truth][2] = counters[truth][2] + 1
                fn += 1
                print(pair)
                if " " in pair[0]:
                    fn_gaps += 1
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
        
        print("fn gaps = ", fn_gaps)
        
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
