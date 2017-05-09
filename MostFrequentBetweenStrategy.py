import numpy as np
import re

class MostFrequentBetweenStrategy:
    
    def __init__(self, nb_feature = 20):
        """
        Constructor
        Initialize all the variables
        """
        self.parent = None
        
        self.nb_feature = nb_feature
        
        self.feature_index = {}
        
        self.feature_words = []
        self.count_words = {}
    

    
    #count specific words in doc list
    def count_words_doc(self, doc_list, nb=-1, verbose = False):
        """
        Given a list of Document object, count the number of each words appearing
        between pairs.
        
        Parameters:
        * nb : used to consider only nb first document of the list
        * verbose : True for debugging output
        """
        nb = len(doc_list) if nb == -1 else nb
        
        for doc in doc_list[:nb]:
            for sentence in doc.sentences:
                if len(sentence.entities) >= 2:

                    all_interaction_text = [p.textBetween for p in sentence.pairs]
                    
                    for interaction_text in all_interaction_text:
                        for w in interaction_text.split(' '):  #for each word of the interaction text list
                            self.count_words[w] = self.count_words.get(w,0) + 1
                    
                    if verbose:
                        print(sentence.text)
                        print(all_interaction_text)
                        print([e.text for e in sentence.entities])
                        print([str(p) for p in sentence.pairs])

                        print("\n-------\n")
    
    
    
    
    def create_feature_word_list(self):
        """
        Using the count of words, compute the list of most used words
        """
        srt = sorted([(self.count_words[w], w) for w in self.count_words], reverse = True)
        
        self.feature_words = [w for (n,w) in srt[:self.nb_feature]]
     
        self.feature_index = {}

        for (i,feat) in enumerate(self.feature_words):
            self.feature_index[feat] = i
    
    
    def get_feature_from_text(self, wordList):
        output = np.zeros(self.nb_feature)
        for w in wordList:
            if w in self.feature_index:
                output[self.feature_index[w]] = 1
        
        return output
    
    def get_features_from_pair(self, pair):
        text = re.split("\W+", pair.textBetween)
        text = " ".join([w for w in text if w != ""])
        
        return self.get_feature_from_text(text)
    
    
    def getFeatureMatrix(self, doc_list, verbose = False):
        """
        Given a list of document
        create the feature matrix of the pairs
        """
        currPair = 0
        
        ncol = self.nb_feature
        nrow = sum([doc_list[d].nbPairs() for d in range(len(doc_list))]) #nb of pairs
        
        matrixFeature = np.zeros(shape=(nrow, ncol))
        
        
        #init : 
        # - count the most occuring words between pairs
        self.count_words_doc(doc_list)
        
        # - create a list of feature words
        self.create_feature_word_list()
        
        
        for doc in doc_list:
            for sentence in doc.sentences:
                if len(sentence.entities) >= 2:
                    
                    for p in sentence.pairs:
                        matrixFeature[currPair] = self.get_features_from_pair(p)
                        currPair += 1
                    
                    if verbose == True:
                        print(sentence.text)
                        print([e.text for e in sentence.entities])
                        print("\n-------\n")
        
        
        return (matrixFeature)