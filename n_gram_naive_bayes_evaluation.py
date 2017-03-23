from corpus_reader import read_dataset
from n_gram_naive_bayes import trained_naive_bayes, extract_features_and_classify
from word_extraction import remove_puntuations
from numpy.random import choice

data = read_dataset()
n_docs = len(data)

train_amount = 0.7
train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
test_ids = [i for i in range(n_docs) if i not in train_ids]

training = [data[i] for i in train_ids]
test = [data[i] for i in test_ids]

print("%i training documents" % len(training))
print("%i test documents" % len(test))

nb = trained_naive_bayes(training)

true_positives = 0
false_positives = 0
false_negatives = 0

single_word_mode = True

for tdoc in test:
    for sentence in tdoc.sentences:
        if len(sentence.text) > 0:
            print("")
            print(sentence.text)
            words = remove_puntuations(sentence.text).split(" ")
            words = [w for w in words if len(w) > 0]
            print("words: " + str(words))
            drugs = []
            last_word_was_drug = False
            for word in words:
                if extract_features_and_classify(word, nb):
                    if last_word_was_drug and not single_word_mode:
                        drugs[-1] = drugs[-1] + " " + word
                    else:
                        drugs.append(word)
                    last_word_was_drug = True
                else:
                    last_word_was_drug = False
            print("prediction: " + str(drugs))
            entity_texts = []
            for entity in sentence.entities:
                entity_texts.append(entity.text)
            print("truth: " + str(entity_texts))
            # evaluate prediction
            for drug in drugs:
                if drug in entity_texts:
                    true_positives = true_positives + 1
                else:
                    false_positives = false_positives + 1
            # evaluate missing drugs
            for truth in entity_texts:
                if truth not in drugs:
                    false_negatives = false_negatives + 1

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
