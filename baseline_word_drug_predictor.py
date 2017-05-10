from pos_tagging import PosTagger

class BaseLineWordDrugPredictor:
    def __init__(self):
        self.drug_names = {}
        self.tagger = PosTagger()
    
    def train(self, data):
        for doc in data:
            for sentence in doc.sentences:
                for entity in sentence.entities:
                    self.drug_names[entity.text] = entity.type
    
    def classify_tagged_words(self, tagged_words):
        labels = ["none"] * len(tagged_words)
        for i, tw in enumerate(tagged_words):
            if tw.word in self.drug_names:
                labels[i] = self.drug_names[tw.word]
        return labels
