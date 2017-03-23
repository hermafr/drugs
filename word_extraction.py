from corpus_reader import read_dataset

#p(c|x) = p(x|c) * p(c)


def overwrite(txt, begin, end, symbol = "#"):
    return txt[0:begin] + symbol * (end - begin) + txt[end:len(txt)]

def remove_puntuations(txt):
    # end of line
    if txt[-1:] == "\n":
        txt = txt[:-1]
    if txt[-1:] == "\r":
        txt = txt[:-1]
    # point in the end
    if txt[-1] == '.':
        txt = txt[:-1]
    # intermediate points and commas
    txt = txt.replace(". ", " ")
    txt = txt.replace(", ", " ")
    return txt

def get_word_lists_from_sentence(sentence):
    txt = sentence.text
    drug_words = []
    neutral_words = []
    if len(txt) > 0:
        for entity in sentence.entities:
            for pos in entity.char_offset:
                txt = overwrite(txt, pos[0], pos[1])
            for word in entity.text.split(" "):
                drug_words.append(word)
        txt = remove_puntuations(txt)
        txt = txt.replace("#", "")
        for word in txt.split(" "):
            if len(word) > 0:
                neutral_words.append(word)
    return drug_words, neutral_words

if __name__ == "__main__":
    data = read_dataset()
    for document in data:
        print("#" * 10)
        print(document.id)
        for sentence in document.sentences:
            drug_words, neutral_words = get_word_lists_from_sentence(sentence)
            print(drug_words)
            print(neutral_words)
