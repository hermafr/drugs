from corpus_reader import read_dataset

data = read_dataset()

drugs = {}

def increase(counter, key):
    counter[key] = 1 if key not in counter else counter[key] + 1

for doc in data:
    for sentence in doc.sentences:
        for entity in sentence.entities:
            increase(drugs, entity.text)

drug_names = sorted(list(drugs))
for drug in drug_names:
    print(drug + "\t" + str(drugs[drug]))
