from pos_tagging import PosTagger
import re

def find_all(string, substring):
    positions = []
    offset = 0
    pos = string.find(substring)
    while pos != -1:
        positions.append(offset + pos)
        offset += pos + 1
        string = string[(pos + 1):]
        pos = string.find(substring)
    return positions

def overlap(i1, i2):
    return not (i1[1] < i2[0] or i1[0] > i2[1])

class BaselineDrugSpanPredictor:
    def __init__(self):
        self.dict = {}
        self.drugs = []
    
    def train(self, data):
        for doc in data:
            for sentence in doc.sentences:
                for entity in sentence.entities:
                    if entity.text != "drug" and entity.text != "drugs":
                        self.dict[entity.text] = entity.type
        entries = [(len(word), word, self.dict[word]) for word in self.dict]
        srt = sorted(entries, reverse = True)
        self.drugs = [(entry[1], entry[2]) for entry in srt]
    
    def classify_spans(self, sentence):
        spans = []
        for drug in self.drugs:
            positions = find_all(sentence, drug[0])
            for pos in positions:
                w_len = len(drug[0])
                new_span = (pos, pos + w_len)
                overlapping = False
                for span in spans:
                    if overlap(span[1][0], new_span):
                        overlapping = True
                        break
                if not overlapping:
                    span = (drug[0],
                            [new_span],
                            drug[1])
                    spans.append(span)
        return spans
