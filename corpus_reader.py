from os import listdir
import xml.etree.ElementTree as etree
import re

corpus_path = "DDICorpus/Train/"
test_task1_path = "DDICorpus/Test/Test for DrugNER task/"
test_task2_path = "DDICorpus/Test/Test for DDI Extraction task/"
folder_names = ["DrugBank/", "MedLine/"]


class Document:
    """ a document has an ID and is a list of sentences
    """
    
    def __init__(self, id):
        """ an empty document
        """
        self.id = id
        self.sentences = []
    
    def add_sentence(self, sentence):
        """ add a sentence to the document
        """
        self.sentences.append(sentence)
    
    def __len__(self):
        """ we define the length of a document as the number of sentences
        """
        return len(self.sentences)
    
    def nbPairs(self):
        """ output the number of total pairs in the document
        """
        return sum( [ len(self.sentences[s].pairs)  for s in range(len(self)) ] )


class Sentence:
    """ a sentence has an ID and consists of text
    it contains a list of entities and a list of pairs
    """
    
    def __init__(self, id, text):
        """ creates a sentence without entities or pairs
        """
        self.id = id
        self.text = text
        self.entities = []
        self.pairs = []
    
    def add_entity(self, entity):
        """ adds an entity to the sentence
        """
        self.entities.append(entity)
    
    def add_pair(self, pair):
        """ adds a pair to the sentence
        """
        self.pairs.append(pair)


class Entity:
    """ an entity represents a drug mention in a sentence
    it has an ID, a type and a text
    its position in the sentence is defined by a character offset
    """
    
    def __init__(self, id, char_offset, type, text):
        """ a new entity
        """
        self.id = id
        self.char_offset = char_offset
        self.type = type
        self.text = text


class Pair:
    """ a pair has an ID and consists of two entities
    it has a boolean ddi and a type iff ddi == true
    """
    
    def __init__(self, id, e1, e2, ddi, type, textBetween, filename=""):
        """ a new pair of drugs
        """
        self.id = id
        self.e1 = e1
        self.e2 = e2
        self.ddi = ddi
        self.type = type
        self.textBetween = textBetween
        self.filename = filename
    
    #when str() is applied to a pair,
    #output a string giving useful information about it
    def __str__(self):
        """ a string containing the stored information
        """
        typeInteraction = ""
    
        if self.ddi == "true":
            if self.type == None: #to avoid special case bug
                typeInteraction = "," + "None"
            else:
                typeInteraction = "," + self.type
        
        return "(" + self.e1.text + "," + self.textBetween +"," +  self.e2.text + "," + self.filename + "," + self.ddi + typeInteraction + ")"

    #gives the label that is the type if interaction else null
    def getLabel(self):
        """ returns the class label or 'null' if no interaction
        """
        label = "null"
    
        if self.ddi == "true":
            label = self.type
            
            if label == None: #to avoid special case bug
                label = "None"
        
        return label

def read_document(file_name):
    """ returns a document instance created from a xml file
    """
    # parse the xml file
    xml = etree.parse(file_name)
    # get the document id
    root = xml.getroot()
    document = Document(root.attrib["id"])
    # loop over sentences in the xml file
    for child in root:
        sentence = Sentence(child.attrib["id"], child.attrib["text"])
        entities = {}
        for element in child:
            if element.tag == "entity":
                offs_str = element.attrib["charOffset"]
                offs = offs_str.split(";")
                char_offset = []
                for off in offs:
                    offs_split = off.split("-")
                    char_offset.append((int(offs_split[0]),
                                        int(offs_split[1]) + 1))
                entity = Entity(element.attrib["id"],
                                char_offset,
                                element.attrib["type"],
                                element.attrib["text"])
                entities[element.attrib["id"]] = entity
                sentence.add_entity(entity)
            elif element.tag == "pair":
                
                e1 = entities[element.attrib["e1"]]
                e2 = entities[element.attrib["e2"]]
                
                text_start = e1.char_offset[0][1] +1
                text_end = e2.char_offset[0][0]        #no +1 because slice in python exclude last end
                
                textBetween = sentence.text[text_start:text_end]    #get the raw text slice
                textBetween = re.split("\W+", textBetween)          #take only words
                textBetween = " ".join([w for w in textBetween if w != ""]) #remove empty words and join the text again
                
                pair = Pair(element.attrib["id"],
                            e1,
                            e2,
                            element.attrib["ddi"],
                            element.attrib["type"] if "type" in element.attrib else None,
                            textBetween,
                            file_name)
                sentence.add_pair(pair)
        document.add_sentence(sentence)
    return document


def read_dataset(test = False, task = 1):
    """ returns the dataset as a list of documents
    test specifies whether the test dataset should be returned (otherwise the training will be used)
    """
    if test:
        if task == 1:
            path = test_task1_path
        else:
            path = test_task2_path
    else:
        path = corpus_path
    documents = []
    for folder_name in folder_names:
        folder_path = path + folder_name
        file_names = listdir(folder_path)
        for file_name in file_names:
            file_path = folder_path + file_name
            documents.append(read_document(file_path))
    return documents
