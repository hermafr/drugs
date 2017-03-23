from corpus_reader import read_dataset
from word_extraction import get_word_lists_from_sentence
from naive_bayes import NaiveBayes

def n_grams(word, n = 3, padding = True):
    if padding:
        pads = "#" * (n - 1)
        word = pads + word + pads
    ngrams = []
    for i in range(0, len(word) - n + 1):
        ngrams.append(word[i:(i+n)])
    return ngrams

def trained_naive_bayes(dataset):
    nb = NaiveBayes([True, False])
    feature_name = "3gram"
    nb.register_feature(feature_name)
    for doc in dataset:
        for s in doc.sentences:
            drug_words, neutral_words = get_word_lists_from_sentence(s)
            for dw in drug_words:
                nb.count_class(True)
                ngrams = n_grams(dw)
                for ngram in ngrams:
                    nb.count_feature(feature_name, ngram, True)
            for nw in neutral_words:
                nb.count_class(False)
                ngrams = n_grams(nw)
                for ngram in ngrams:
                    nb.count_feature(feature_name, ngram, False)
    return nb

def extract_features_and_classify(word, nb):
    ngrams = n_grams(word)
    features = []
    for ngram in ngrams:
        features.append(("3gram", ngram))
    label = nb.classify(features)
    return label

if __name__ == "__main__":
    dataset = read_dataset()
    nb = trained_naive_bayes(dataset)
    while True:
        print("enter example:")
        example = input()
        label = extract_features_and_classify(example)
        print(label)
