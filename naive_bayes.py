
VERBOSE = False

EPSILON_SMOOTHING = True
EPSILON = 0.000001
ALPHA = 1

def verbose(str):
    if VERBOSE:
        print(str)

class NaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.features = {}
        self.f_counters = {}
        self.n = 0
        self.class_frequencies = {}
        for class_label in classes:
            self.class_frequencies[class_label] = 0
    
    def count_class(self, class_label):
        self.n = self.n + 1
        self.class_frequencies[class_label] = self.class_frequencies[class_label] + 1
    
    def register_feature(self, feature_name):
        if feature_name not in self.features:
            self.features[feature_name] = {}
            self.f_counters[feature_name] = {}
            for c in self.classes:
                self.f_counters[feature_name][c] = 0
    
    def count_feature(self, feature_name, value, class_label):
        if value not in self.features[feature_name]:
            self.features[feature_name][value] = {}
            for c in self.classes:
                self.features[feature_name][value][c] = 0
        self.features[feature_name][value][class_label] = self.features[feature_name][value][class_label] + 1
        self.f_counters[feature_name][class_label] = self.f_counters[feature_name][class_label] + 1
    
    def p_class(self, c):
        if c not in self.class_frequencies:
            freq = 0
        else:
            freq = self.class_frequencies[c]
        freq = freq
        p = freq / self.n
        p = p + EPSILON
        verbose("class\t" + str(c) + "\t" + str(freq) + "\t" + str(self.n) + "\t" + str(p))
        return p
    
    def p_feature_given_class(self, feature_name, value, class_label):
        if value not in self.features[feature_name]:
            freq = 0
        else:
            freq = self.features[feature_name][value][class_label]
        freq = freq
        divisor = self.f_counters[feature_name][class_label]
        if EPSILON_SMOOTHING:
            p = freq / divisor
            p = p + EPSILON
        else:
            p = (freq + ALPHA) / (divisor + ALPHA * len(self.features[feature_name]))
        verbose(feature_name + "\t" + str(value) + "\t" + str(freq) + "\t" + str(divisor) + "\t" + str(p))
        return p
    
    def get_prob(self, features, c, use_prior):
        verbose(c)
        verbose("-" * 10)
        p = 1
        if use_prior:
            p = self.p_class(c)
        for f in features:
            fname = f[0]
            fvalue = f[1]
            p = p * self.p_feature_given_class(fname, fvalue, c)
            verbose("\t" + str(p))
        return p
    
    def classify(self, features, use_prior=True):
        max_p = -1
        predicted_class = None
        for c in self.classes:
            p = self.get_prob(features, c, use_prior)
            verbose("-" * 10)
            verbose("p(" + str(c) + ") = " + str(p))
            verbose("")
            if p > max_p:
                max_p = p
                predicted_class = c
        return predicted_class
