from corpus_reader import read_dataset

def increase(counter, key):
    counter[key] = 1 if key not in counter else counter[key] + 1

def get_drug_frequencies():
    data = read_dataset()
    drugs = {}
    for doc in data:
        for sentence in doc.sentences:
            for entity in sentence.entities:
                increase(drugs, entity.text)
    return drugs

if __name__ == "__main__":
    drugs = get_drug_frequencies()
    drug_names = sorted(list(drugs))
    for drug in drug_names:
        print(drug + "\t" + str(drugs[drug]))
