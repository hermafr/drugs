from corpus_reader import read_dataset

data = read_dataset()

for doc in data:
    for s in doc.sentences:
        print(s.text)
        for p in s.pairs:
            print(p.e1.text, p.e2.text, p.type)
        print("")
