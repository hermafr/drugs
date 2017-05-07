from corpus_reader import read_dataset
#from n_gram_naive_bayes import trained_naive_bayes, extract_features_and_classify

from word_extraction import remove_puntuations
from numpy.random import choice

from sklearn import svm

data = read_dataset()
n_docs = len(data)

train_amount = 0.7
train_ids = choice(n_docs, int(train_amount * n_docs), replace=False)
test_ids = [i for i in range(n_docs) if i not in train_ids]

training = [data[i] for i in train_ids]
test = [data[i] for i in test_ids]

print("%i training documents" % len(training))
print("%i test documents" % len(test))

clf = svm.SVC(probability=True)

#training is list of document
#a document contains sentences
#in sentences we have list of  pairs
#pairs have
#   e1      entity 1
#   e2      entity 2
#   ddi     = true if they interact
#   type    type of interaction if ddi=true

"""

For each sentences, keep it if number of drugs > 2

"""

nb = 1


nbMayInteractSentence = 0
nbSentence = 0

for doc in training[0:nb]:
    for sentence in doc.sentences:
        nbSentence += 1
        if len(sentence.entities) >= 2:
            nbMayInteractSentence += 1
            

print str(nbMayInteractSentence) + " out of " + str(nbSentence) + " sentences ("+ str(round(10000*float(nbMayInteractSentence)/nbSentence)/100) +"%) have 2 drugs or more."


####

def get_pairs_interaction(pair):
    return "(" + pair.e1.text + "," +  pair.e2.text + "," + pair.ddi + ")"

    
def get_interaction_text(pair, sentenceText):
    print (pair.e1.char_offset)
    #text_start = pair.e1.char_offset[1]
    #text_end = pair.e2.char_offset[0]
    
    print "TEXT START AT " + str(text_start) + "AND ENDS AT " + str(text_end) + "\n"
    
    #text = sentenceText[text_start:text_end]
    text = "--"
    return "(" + pair.e1.text + "," + text + "," +  pair.e2.text + "," + pair.ddi + ")"

nb=1

for doc in training[0:nb]:
    for sentence in doc.sentences:
        nbSentence += 1
        if len(sentence.entities) >= 2:
            print(sentence.text)
            print([e.text for e in sentence.entities])
            #print([get_pairs_interaction(p) for p in sentence.pairs])
            print([get_interaction_text(p, sentence.text) for p in sentence.pairs])
            print("\n-------\n")
            





nbInteract = 0
total = 0
for doc in training[0:nb]:
    for sentence in doc.sentences:
        for pairs in sentence.pairs:
            total += 1
            if pairs.ddi == "true":
                #print pairs.e1 + " interacts with " + pairs.e2 + " and it is type '" + pairs.type + "'" 
                nbInteract += 1
                
                



print str(nbInteract) + " out of " + str(total) + " drugs ("+ str(round(10000*float(nbInteract)/total)/100) +"%) are interacting."

#instead of having pair entities id, better have name ?
#
#given a sentence, is it mentioned drugs that interact ?


#Task: Extraction and classification of drugs interactions
#
# - Extraction: given a sentence, find the interactions
# - Classification: give the type of found interaction ?


#clf.fit(training,labels_train.T[i])