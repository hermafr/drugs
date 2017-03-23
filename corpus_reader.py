from os import listdir
import xml.etree.ElementTree as etree


corpus_path = "DDICorpus/Train/"
folder_names = ["DrugBank/", "MedLine/"]


# a document has an ID and is a list of sentences
class Document:
    def __init__(self, id):
        self.id = id
        self.sentences = []
    
    def add_sentence(self, sentence):
        self.sentences.append(sentence)


# a document has an ID and consists of text
# it contains a list of entities and a list of pairs
class Sentence:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.entities = []
        self.pairs = []
    
    def add_entity(self, entity):
        self.entities.append(entity)
    
    def add_pair(self, pair):
        self.pairs.append(pair)


# an entity has an ID, a type and a text
# its position in the sentence is defined by a character offset
class Entity:
    def __init__(self, id, char_offset, type, text):
        self.id = id
        self.char_offset = char_offset
        self.type = type
        self.text = text


# a pair has an ID and consists of two entity IDs
# it has a boolean ddi and a type iff ddi == true
class Pair:
    def __init__(self, id, e1, e2, ddi, type):
        self.id = id
        self.e1 = e1
        self.e2 = e2
        self.ddi = ddi
        self.type = type


# returns a document instance created from a xml file
def read_document(file_name):
    # parse the xml file
    xml = etree.parse(file_name)
    # get the document id
    root = xml.getroot()
    document = Document(root.attrib["id"])
    # loop over sentences in the xml file
    for child in root:
        sentence = Sentence(child.attrib["id"], child.attrib["text"])
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
                sentence.add_entity(entity)
            elif element.tag == "pair":
                pair = Pair(element.attrib["id"],
                            element.attrib["e1"],
                            element.attrib["e2"],
                            element.attrib["ddi"],
                            element.attrib["type"] if "type" in element.attrib else None)
                sentence.add_pair(pair)
        document.add_sentence(sentence)
    return document


def read_dataset():
    documents = []
    for folder_name in folder_names:
        path = corpus_path + folder_name
        file_names = listdir(path)
        for file_name in file_names:
            file_path = path + file_name
            documents.append(read_document(file_path))
    return documents
