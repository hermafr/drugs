import lib.freeling as freeling


class FreebaseAdapter:
    def __init__(self):
        # set locale to an UTF8 compatible locale 
        freeling.util_init_locale("default");
        # get requested language
        self.lang = "en"
        # get installation path
        self.ipath = "/usr";
        # path to language data   
        self.lpath = self.ipath + "/share/freeling/" + self.lang + "/"
    
    def my_maco_options(self, lang,lpath) :
        # create options holder 
        opt = freeling.maco_options(self.lang);
        # Provide files for morphological submodules. Note that it is not 
        # necessary to set file for modules that will not be used.
        opt.UserMapFile = "";
        opt.LocutionsFile = self.lpath + "locucions.dat"; 
        opt.AffixFile = self.lpath + "afixos.dat";
        opt.ProbabilityFile = self.lpath + "probabilitats.dat"; 
        opt.DictionaryFile = self.lpath + "dicc.src";
        opt.NPdataFile = self.lpath + "np.dat"; 
        opt.PunctuationFile = self.lpath + "../common/punct.dat"; 
        return opt;
    
    def tokenizer(self):
        return freeling.tokenizer(self.lpath+"tokenizer.dat");
    
    def splitter(self):
        return freeling.splitter(self.lpath+"splitter.dat");
    
    def morfo(self):
        morfo=freeling.maco(self.my_maco_options(self.lang,self.lpath));
        morfo.set_active_options (False,  # UserMap 
                                  True,  # NumbersDetection,  
                                  True,  # PunctuationDetection,   
                                  True,  # DatesDetection,    
                                  True,  # DictionarySearch,  
                                  True,  # AffixAnalysis,  
                                  False, # CompoundAnalysis, 
                                  True,  # RetokContractions,
                                  True,  # MultiwordsDetection,  
                                  True,  # NERecognition,     
                                  False, # QuantitiesDetection,  
                                  True); # ProbabilityAssignment
        return morfo
    
    def tagger(self):
        return freeling.hmm_tagger(self.lpath+"tagger.dat",True,2)
