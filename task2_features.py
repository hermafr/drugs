import numpy as np

clue_words = ["no",
              "not",
              "should",
              "increase",
              "decrease"
              ]


types = ["null",
         "effect",
         "advise",
         "int",
         "mechanism"
         ]

def get_num_features():
    return len(clue_words) + len(types)

def get_features(pair, sentence):
    f = np.zeros(get_num_features())
    i = 0
    for word in clue_words:
        if word in sentence.text:
            f[i] = 1
        i += 1
    for t in types:
        if pair.type == t:
            f[i] = 1
        i += 1
    return f
