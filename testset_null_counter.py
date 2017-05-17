from corpus_reader import read_dataset

data = read_dataset(test = True, task = 2)

nulls = 0
n = 0

for doc in data:
    for sentence in doc.sentences:
        for pair in sentence.pairs:
            n += 1
            if pair.type == None:
                nulls += 1

print("n", n)
print("nulls", nulls)
print("%", nulls / n)
