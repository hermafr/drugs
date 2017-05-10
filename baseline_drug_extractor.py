from corpus_reader import read_dataset
from pos_ngram_naive_bayes import get_spans_from_entities, get_texts_from_entities
from numpy.random import choice

VERBOSE = True

def get_substring_beginnings(string, pattern):
    positions = []
    pos = 0
    while True:
        add = string[pos:].find(pattern)
        if add == -1:
            return positions
        pos = pos + add
        positions.append(pos)
        pos = pos + 1


class BaselineDrugClassifier:
    def __init__(self):
        self.drugdict = set()
    
    def train(self, training):
        for doc in training:
            for sentence in doc.sentences:
                for entity in sentence.entities:
                    self.drugdict.add(entity.text)


if __name__ == "__main__":
    data = read_dataset()
    n_docs = len(data)
    train_amount = 0.7
    train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
    test_ids = [i for i in range(n_docs) if i not in train_ids]
    training = [data[i] for i in train_ids]
    test = [data[i] for i in test_ids]

    baseline = BaselineDrugClassifier()
    baseline.train(training)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for doc in test:
        for sentence in doc.sentences:
            predicted_spans = []
            predicted_words = []
            for drug in baseline.drugdict:
                starts = get_substring_beginnings(sentence.text, drug)
                for start in starts:
                    predicted_spans.append([(start, start + len(drug))])
                    predicted_words.append(drug)
            drug_spans = get_spans_from_entities(sentence.entities)
            # concatenate predictions:
            # predicted_words, predicted_spans = simple_concatenation(predicted_words, predicted_spans)
            # evaluate:
            for predicted in predicted_spans:
                if predicted in drug_spans:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
            for target in drug_spans:
                if target not in predicted_spans:
                    false_negatives = false_negatives + 1
            # output:
            if VERBOSE:
                print("")
                print("sentence: %s" % sentence.text)
                print("predictions: %s" % str(predicted_words))
                print("truth: %s" % str(get_texts_from_entities(sentence.entities)))
    
    
    print("")
    print("-" * 10)
    print("%i true positives" % true_positives)
    print("%i false positives" % false_positives)
    print("%i false negatives" % false_negatives)
    print("")

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print("recall: %f" % recall)
    print("precision: %f" % precision)
