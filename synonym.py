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
                                senses_synonym_words = []
                                for synonym in senses.find_all('SenseExample'):
                                    synonym_ = synonym.find('feat',{'att':'word'})['val']
                                    rank = synonym.find('feat',{'att':'rank'})['val']  
                                    usage = senses.find('feat',{'att':'usage'})['val']
                                    senses_synonym_words.append((synonym_,int(rank)))                            
                                self.sort_store_synonyms(word,senses_synonym_words,postag,usage,state=True)
                                print()
                                print()          
                        else:
                            one_sense_synonym_words = []
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

        input: word, list of synonyms
        output: sorted list of synonyms 
        """

        synonym_list = []
        sorted_synonyms = sorted(synonyms, key = lambda x:x[1], reverse = False)
        for values in sorted_synonyms:
            synonym_list.append(values[0])
        index = synonym_list.index(word)
        if synonym_list.index(word) == 0:
            if state == False:
                self.resyf.append({'word':word,'pos':pos,'synonym_rank':sorted_synonyms[index]})
            else:
                self.resyf.append({'word':word,'pos':pos,'usage':usage,'synonym_rank':sorted_synonyms[index]})
        else:
            if state == False:
                self.resyf.append({'word':word,'pos':pos,'synonym_rank':sorted_synonyms[:index]})
            else:
                self.resyf.append({'word':word,'pos':pos,'usage':usage,'synonym_rank':sorted_synonyms[:index]})
   

    def get_synonym(self,word,pos):
        """
        Provides sets of synonyms

        input: word, part of speech (pos)
        output: synonym with associated rank of the inserted word.
        """
        for record in self.resyf:
            if record['word'] == word and record['pos']==pos:
                    return record['synonym_rank']

                
            
                
