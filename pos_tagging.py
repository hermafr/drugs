from freebase_adapter import FreebaseAdapter

class TaggedWord:
    """ represents a tagged word
    """
    def __init__(self, word, lemma, span, pos):
        self.word = word
        self.lemma = lemma
        self.span = span
        self.pos = pos
    
    def __str__(self):
        """ prints a tuple with all tags
        """
        return str((self.word, self.lemma, self.span, self.pos))

class PosTagger:
    """ adds POS tags to sentences
    """
    
    def __init__(self):
        """ creates objects needed from freeling
        """
        adapter = FreebaseAdapter()
        
        # create analyzers
        self.tk=adapter.tokenizer();
        self.sp=adapter.splitter();

        # create the analyzer with the required set of maco_options  
        self.morfo = adapter.morfo();
        
        # create tagger
        self.tagger = adapter.tagger()

    def pos_tag(self, sentence):
        """ returns a list of tagged words, given a sentence
        """
        # tokenize input line into a list of words
        lw = self.tk.tokenize(sentence)
        # split list of words in sentences, return list of sentences
        ls = self.sp.split(lw)

        # perform morphosyntactic analysis and disambiguation
        ls = self.morfo.analyze(ls)
        ls = self.tagger.analyze(ls)
        
        # create (word, POS-tag) pairs for each word:
        tagged_words = []
        
        # for each sentence in list
        for s in ls :
            # for each word in sentence
            for w in s :
                # print word form  
                word = w.get_form()
                lemma = w.get_lemma()
                span = (w.get_span_start(), w.get_span_finish())
                tag = w.get_tag()
                tagged_words.append(TaggedWord(word, lemma, span, tag))
        return tagged_words

if __name__ == "__main__":
    tagger = PosTagger()
    while True:
        print("")
        print(">> enther phrase:")
        text = input()
        tagged_words = tagger.pos_tag(text)
        for tw in tagged_words:
            print(str(tw))
