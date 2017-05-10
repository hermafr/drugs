from corpus_reader import read_dataset
import numpy as np

class FeatureReader:
    def __init__(self):
        self.lca_features = {}
        self.read_lca_features()
    
    def read_lca_features(self):
        filename = "results/pairs_lowest_ancestor_features.txt"
        with open(filename) as f:
            lines = f.readlines()
        for line in lines:
            line = line[:-1]
            values = line.split(",")
            self.lca_features[values[0]] = (values[2], ",".join(values[3:]))
    
    def get_lca_tag(self, pair_id):
        return self.lca_features[pair_id][0]
    
    def get_lca_word(self, pair_id):
        return self.lca_features[pair_id][1]


if __name__ == "__main__":
    fr = FeatureReader()
    data = read_dataset()
    
    counters = {}
    
    for doc in data:
        for sentence in doc.sentences:
            for pair in sentence.pairs:
                #feature = fr.get_lca_tag(pair.id)
                feature = fr.get_lca_word(pair.id)
                c = pair.type if pair.type != None else "none"
                ## reverse counting
                if True:
                    cc = c
                    c = feature
                    feature = cc
                ##
                if feature not in counters:
                    counters[feature] = {}
                if c not in counters[feature]:
                    counters[feature][c] = 0
                counters[feature][c] = counters[feature][c] + 1
    
    occurences = []
    for feature in counters:
        num_occs = 0
        for c in counters[feature]:
            num_occs += counters[feature][c]
        occurences.append((num_occs, feature))
    srt = [val[1] for val in sorted(occurences, reverse=True)]
    
    for feature in srt:
        print("%s : %s" % (feature, counters[feature]))
