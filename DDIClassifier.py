from time import time
import numpy as np

class DDIClassifier:

    def __init__(self, featureStrategy, clf):
        """
        Constructor
        Initialize all the variables
        """
        
        self.classes = ['null', 'advise', 'effect', 'int', 'mechanism']
        #given a class label, gives the corresponding value
        self.class_index = {self.classes[i] : i for (i,e) in enumerate(self.classes)}
        
        #given a class value, gives the corresponding class label
        self.class_mapping = {v: k for k, v in self.class_index.items()}
        
        self.clf = clf
        
        self.featureStrategy = featureStrategy
    
    def getFeatureMatrix(self, doc_list):
        return self.featureStrategy.getFeatureMatrix(doc_list)
    
    def getLabels(self, doc_list):
        """
        Given a document list
        gives the true labels
        """
        labels = []
        
        for doc in doc_list:
            for sentence in doc.sentences:
                if len(sentence.entities) >= 2:
                    for p in sentence.pairs:
                        labels.append(self.class_index[p.getLabel()])
        
        
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
        
        return self.class_mapping[self.clf.predict(f)[0]]
    
    def predict(self, matrix):
        """
        Given a matrix,
        predicts
        """
        