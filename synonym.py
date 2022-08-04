from bs4 import BeautifulSoup


class synonyms:
    def __init__(self,xml_file):
        
        self.vocab_synonym = list()
        with open(xml_file, 'r') as f:
            data = f.read()
            Bs_data = BeautifulSoup(data, "xml")
            for tag in Bs_data.find_all('LexicalEntry'):
                word = tag.find('feat',{'att':'lexeme'})['val']
                postag = tag.find('feat',{'att':'partOfSpeech'})['val']
                synonyms = set()
                if len(word.split(' ')) == 1:
                    for senses in tag.find_all('Sense'):
                        if senses.find('feat',{'att':'poids'}) is not None:
                          sense = senses.find('feat',{'att':'poids'})['val']
                          usage = senses.find('feat',{'att':'usage'})['val']
                        else:
                          sense = '0'
                          usage = ''
                        if float(sense) > 0.99:
                            for synonym in senses.find_all('SenseExample'):
                                synonym_ = synonym.find('feat',{'att':'word'})['val']
                                rank = synonym.find('feat',{'att':'rank'})['val']  
                                if int(rank) == 1: 
                                    synonyms.add(synonym_)
                                self.vocab_synonym.append({'word':word,'usage':usage, 'sense':sense,'pos':postag, 'synonyms':synonyms})

    
    def get_synonyms(self,word,tag, synonym_usage=False):
        synonym = ''
        for index in range(len(self.vocab_synonym)):
            if self.vocab_synonym[index]['word'] == word and self.vocab_synonym[index]['pos']==tag:        
                    synonym = list(self.vocab_synonym[index]['synonyms'])[0]
                    break
        return synonym
            
            
                
