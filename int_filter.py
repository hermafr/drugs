from corpus_reader import read_dataset

data = read_dataset
for doc in data:
    for sentence in doc.sentences:
        for pair in sentence.pairs:
            if 
