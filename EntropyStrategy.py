import numpy as np
import re

class EntropyStrategy:
    """
    In this strategy, we count the words that are occuring between pairs.
    Since some words may occur everywhere for all classes, there are not useful
    Then, we calculate the entropy of each counted word for each class
    """
    
    def __init__(self, parent, nb_feature = 20, threshold_count = 20):
        """
        Constructor
        Initialize all the variables
        """

        
        self.parent = parent
        self.classes = parent.classes
        
        self.nb_feature = nb_feature
        self.threshold_count = threshold_count
        
        
        self.feature_index = {}
        
        self.feature_words = []
        self.count_words = {} #each words has its number of occurence in each class
    
        
    
    
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
        print("COUNT WORDS")
        for doc in doc_list[:nb]:
            for sentence in doc.sentences:
                if len(sentence.entities) >= 2:

                    for p in sentence.pairs:
                        for w in p.textBetween.split(' '): #for each for of the text between
                            w = w.lower()
                            if w not in self.count_words: #if new word
                                self.count_words[w] = {c : 0 for c in self.classes}
                                

                            p_class = self.parent.labelToClass[p.getLabel()]
                            
                            self.count_words[w][p_class] = self.count_words[w][p_class] + 1
                    
                    
                    if verbose:
                        print(sentence.text)
                        print([e.text for e in sentence.entities])
                        print([str(p) for p in sentence.pairs])

                        print("\n-------\n")
    
    
    def filter_word(self, word_classes):
        """
        Given a word occurences for classes
        Filter it
        Criteria : filter if  nb of occurences of the words < self.threshold_count
        """
        
        return sum([word_classes[k] for k in word_classes]) < self.threshold_count
    
    def get_probability_count_words(self):
        output = {}
        for w in self.count_words.keys():  #for each word
            if not self.filter_word(self.count_words[w]): #consider the word only if not filtered
                dict = self.count_words[w]
                total = float(sum([dict[k] for k in dict]))
                
                output[w] = {w : (n/total) for (w, n) in dict.items()}
 
        return output
 
    def entropy(self, p_classes):
        """
        Given a probability p_classes for each class
        Returns the entropy
        """
        
        H = 0
        
        for c in p_classes:
            p = p_classes[c]
            H += p * np.log(p) if p != 0 else 0
        
        return -H

    def get_entropy(self, p_count_words):
        """
        Given the probability of each word for each class
        Returns the entropy of each words
        """
        output = {}
        
        for w in p_count_words: #for each word
            output[w] = self.entropy(p_count_words[w])
        
        return output
 
    def create_feature_word_list(self, entropyDict):
        """
        Using the count of words, compute the list of most used words
        """
        srt = sorted([(entropyDict[w], w) for w in entropyDict])
        
        self.feature_words = [w for (n,w) in srt[:self.nb_feature]]
     
        self.feature_index = {}

        for (i,feat) in enumerate(self.feature_words):
            self.feature_index[feat] = i
    
    
    def get_feature_from_text(self, wordList):
        '''
        Given a list of words
        Transform it as a feature vector
        '''
        output = np.zeros(self.nb_feature)
        for w in wordList:
            if w in self.feature_index:
                output[self.feature_index[w]] = 1
        
        return output
    
    def get_features_from_pair(self, pair):
        '''
        Given a pair, get the text between the two drugs involved
        Then create a feature vector calling another method
        '''
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
        
        # - count the probability of these words for each class
        p = self.get_probability_count_words()
        
        # - calculate the entropy of each words
        entropy = self.get_entropy(p)
        
        # - create a list of feature words
        self.create_feature_word_list(entropy)
        
        
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