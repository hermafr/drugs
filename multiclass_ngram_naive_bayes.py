from naive_bayes import NaiveBayes
from pos_tagging import PosTagger, TaggedWord
from feature_computation import n_grams, starts_with_uppercase
from corpus_reader import read_dataset
from numpy.random import choice

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
            labels.append(label)
        return labels
    
    def classify_tagged_words(self, tagged_words):
        labels = []
        feature_list = self.tagged_words_to_features(tagged_words)
        for i in range(len(tagged_words)):
            label = self.nb.classify(feature_list[i])
            labels.append(label)
        return labels

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
            span = [tagged_words[i].span]
            for j in range(n_entities):
                if span == entities[j].char_offset:
                    pairs.append((tagged_words[i].word, entities[j].type, labels[i]))
                    entities_classified_as[j] = labels[i]
    # get missing predictions
    for i in range(n_entities):
        if entities_classified_as[i] == "none":
            pairs.append((entities[i].text, entities[i].type, entities_classified_as[i]))
    return pairs

if __name__ == "__main__":
    data = read_dataset()
    n_docs = len(data)
    train_amount = 0.7
    train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
    test_ids = [i for i in range(n_docs) if i not in train_ids]
    training = [data[i] for i in train_ids]
    test = [data[i] for i in test_ids]
    
    print("%i training documents" % len(training))
    print("%i test documents" % len(test))
    
    nb = MulticlassPosNgramNaiveBayes()
    nb.train(training)
    
    counters = {}
    for c in nb.classes:
        if c != "none":
            counters[c] = [0, 0, 0]  # true positives, false positive, false negatives
    
    for doc in test:
        for sentence in doc.sentences:
            tagged_words = nb.tagger.pos_tag(sentence.text)
            labels = nb.classify_tagged_words(tagged_words)
            gt_pred_pairs = get_gt_prediction_pairs(tagged_words, sentence.entities, labels)
            
            if VERBOSE:
                print(sentence.text)
                print(gt_pred_pairs)
                print("")
            
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
    
    for c in counters:
        true_positives = counters[c][0]
        false_positives = counters[c][1]
        false_negatives = counters[c][2]
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print("%s: %s, p=%f, r=%f" % (c, str(counters[c]), precision, recall))
