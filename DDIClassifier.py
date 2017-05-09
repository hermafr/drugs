from time import time
import numpy as np
import sys

class DDIClassifier:

    def __init__(self, classes, featureStrategy, clf, mappingLabelToClass = None):
        """
        Constructor
        Initialize all the variables
        """
        
        self.classes = classes
        self.labelsList = ['null', 'advise', 'effect', 'int', 'mechanism']
        
        
        if mappingLabelToClass == None or classes == self.labelsList:
            self.labelToClass = {l : l for l in self.labelsList}

            
        else:
            for label in self.labelsList:   #for each label, check if is in the mapping
                if not label in mappingLabelToClass:
                    sys.exit("Error, mapping label->class incorrect")
                
            
            self.labelToClass = mappingLabelToClass
            
        #given a class label, gives the corresponding value
        self.class_index = {self.classes[i] : i for (i,e) in enumerate(self.classes)}
        
        #given a class value, gives the corresponding class label
        self.class_mapping = {v: k for k, v in self.class_index.items()}
        
        self.clf = clf
        
        self.clf = clf
        
        if featureStrategy["name"] == "entropy":
            from EntropyStrategy import EntropyStrategy
            thresh = featureStrategy["threshold_count"] if "threshold_count" in featureStrategy else 30
            nb_feat = featureStrategy["nb_feature"] if "nb_feature" in featureStrategy else 1000
            self.featureStrategy = EntropyStrategy(self, nb_feature = nb_feat, threshold_count = thresh)
        
    
    def mapLabelToClass(self,label):
        return "no_interaction" if label == "null" else "interaction"    
    
    def getFeatureMatrix(self, doc_list):
        return self.featureStrategy.getFeatureMatrix(doc_list)
    
    def getClasses(self, doc_list):
        """
        Given a document list
        gives the true classes
        """
        labels = []
        
        for doc in doc_list:
            for sentence in doc.sentences:
                if len(sentence.entities) >= 2:
                    for p in sentence.pairs:
                        p_class = self.labelToClass[p.getLabel()]
                        labels.append(self.class_index[p_class])
        
        
        return np.array(labels)
                        
    
    def fit(self, trainingFeature, labels, verbose=False):
        if verbose:
            print("Start training")
        start = time()
        
        self.clf.fit(trainingFeature,labels)
        
        if verbose:
            print ("Done in " + str(time() - start) + "s")

    
    def predictFromTextBetween(self, textBetween):
        s = textBetween.lower().split(' ')
        f = self.featureStrategy.get_feature_from_text(s).reshape(1,-1)
        
        pred = self.clf.predict(f)[0]
        
        
        
        return self.class_mapping[pred]
    
    #def predict(self, matrix):
    #    """
    #    Given a matrix,
    #    predicts
    #    """
    
    
    #SVC (but not NuSVC) implement a keyword class_weight in the fit method.
    #It is a dictionary of the form {class_label : value}, where value is a floating point number > 0
    #as it is unbalanced, balance the classes
    def weight_balancing(self, classesList):
        """Given the list of classes
        returns the weight to balance evenly the classes
        """
        
        print(len(classesList))
        
        c = {}
        for l in classesList:
            c[l] = c.get(l,0) + 1
        
        weight = {i: len(classesList)/float(c[i]) for i in c.keys()}
        
        return weight