from corpus_reader import read_dataset
from word_extraction import get_word_lists_from_sentence
from feature_computation import n_grams

data = read_dataset()

counters = {}

for doc in data:
    for sentence in doc.sentences:
        drug_words, neutral_words = get_word_lists_from_sentence(sentence)
        for word in drug_words:
            if len(word) > 0:
                ngrams = n_grams(word)
                for ngram in ngrams:
                    if ngram not in counters:
                        counters[ngram] = [0, 0]
                    counters[ngram][1] = counters[ngram][1] + 1
        for word in neutral_words:
            if len(word) > 0:
                ngrams = n_grams(word)
                for ngram in ngrams:
                    if ngram not in counters:
                        counters[ngram] = [0, 0]
                    counters[ngram][0] = counters[ngram][0] + 1

lst = [(counters[ngram][0], counters[ngram][1], ngram) for ngram in counters]
srt = sorted(lst, reverse = True)

for elem in srt:
    if "\n" not in elem[2]:
        print("%s,%i,%i" % (elem[2], elem[0], elem[1]))
