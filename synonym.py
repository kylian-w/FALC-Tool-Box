from bs4 import BeautifulSoup

class resyf:
    def __init__(self):
        self.resyf = []
    def load(self,xml_file):
        """
        Loads the ontology from xml file

        input: xml file
        output: None
        """
        one_sense_synonym_words = []
        senses_synonym_words = []
        with open(xml_file, 'r') as f:
            data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            for tag in Bs_data.find_all('LexicalEntry'):
                word = tag.find('feat',{'att':'lexeme'})['val']
                postag = tag.find('feat',{'att':'partOfSpeech'})['val']
                if len(word.split(' ')) == 1: ##Working with unigrams
                    for senses in tag.find_all('Sense'):
                        if senses.find('feat',{'att':'poids'}) is not None:
                            sense = senses.find('feat',{'att':'poids'})['val']
                            if float(sense) == 1.0:
                                senses_synonym_words = [] ##managing synonyms with several senses
                                for synonym in senses.find_all('SenseExample'):
                                    synonym_ = synonym.find('feat',{'att':'word'})['val']
                                    rank = synonym.find('feat',{'att':'rank'})['val']  
                                    usage = senses.find('feat',{'att':'usage'})['val']
                                    senses_synonym_words.append((synonym_,int(rank)))                            
                                self.sort_store_synonyms(word,senses_synonym_words,postag,usage,state=True)     
                        else:
                            one_sense_synonym_words = [] ##managing one sense synonyms
                            for synonym in senses.find_all('SenseExample'):
                                synonym_ = synonym.find('feat',{'att':'word'})['val']
                                rank = synonym.find('feat',{'att':'rank'})['val'] 
                                one_sense_synonym_words.append((synonym_, int(rank)))
                            self.sort_store_synonyms(word,one_sense_synonym_words,postag)

            print('Done loading the ontology.')

    def sort_store_synonyms(self,word,synonyms,pos,usage='',state=False):
        """
        Filters the list of synonyms according to their ranks
        and position of the word in the list 

        input: 
            - word: identified word
            - synonyms: list of synonyms of the identified word
            - pos: part of speech
            - usage: usage of the word in a corpus
            - state: determines whether to put the usage of the word in the corpus in the final result
        output: sorted list of synonyms 
        """

        synonym_list = []
        sorted_synonyms = sorted(synonyms, key = lambda x:x[1], reverse = False) ## sorting list of synonyms with asssociated rank with synonym
        for values in sorted_synonyms:
            synonym_list.append(values[0]) ## extracting the words from the list for further filtering within the list.
        index = synonym_list.index(word)
        if synonym_list.index(word) == 0:
            if state == False:## scenario where we fall on the synonym itself
                self.resyf.append({'word':word,'pos':pos,'synonym_rank':[sorted_synonyms[index]]})
            else:
                self.resyf.append({'word':word,'pos':pos,'usage':usage,'synonym_rank':[sorted_synonyms[index]]})
        else:## scenario where we have several occurence of synonyms
            if state == False:
                self.resyf.append({'word':word,'pos':pos,'synonym_rank':sorted_synonyms[:index]})
            else:
                self.resyf.append({'word':word,'pos':pos,'usage':usage,'synonym_rank':sorted_synonyms[:index]})
   

    def get_synonym(self,word,pos):
        """
        Provides sets of synonyms

        input: 
            - word: complex word
            - pos: part of speech 

        output: synonym with associated rank of the inserted word.
        """
        try:
            for record in self.resyf:
                if record['word'] == word.lower() and record['pos']==pos.upper():
                        return [word,record['synonym_rank']]
        except SyntaxError:
            print('The input data should be a text')
    
    def get_all_synonyms(self,words):
        """
        Gets the synonyms of several words at a time

        input: 
            - list of key values of words

        output: list of words and their associated synonyms
        """
        ##TODO: Handeling scenario where words is not in the form of a list
        try:
            word_synonyms = []
            for record in words:
                if self.get_synonym(record[0].lower(),record[1][1].upper()) is not None:
                    word_synonyms.append(self.get_synonym(record[0].lower(),record[1][1].upper()))
                else:
                    word_synonyms.append(self.get_synonym(record[1][0].lower(),record[1][1].upper()))
            return word_synonyms
        except TypeError:
            print('Your input data should be in the format [word,[(synonym, rank)]]')
                
            
                
