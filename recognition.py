# -*- coding: utf-8 -*-

import PyPDF2
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
import pickle
import time
import datetime
import stanza


class preprocess:
  """
  NLP preprocessing by Engineers at ISEP

  """
  def __init__(self):
      self.sentence_token = list()
      self.total_token = 0
      self.total_corpus_token = 0

  def extract_text_pdf(self,pdf_directory):
      """
      This function collects text from a pdf and outputs a
      processed list of the text.

      input: pdf direcotry
      output: list of preprocessed text
      """
      self.total_token = 0
      pdf =  open(pdf_directory,'rb')
      pdf = PyPDF2.PdfFileReader(pdf)
      print('Document ', pdf_directory)
      for i in range(pdf.numPages):
        print('Page ',i+1)
        pagepdf = pdf.getPage(i)
        _ , total = self.preprocess_text([self.lower_text(pagepdf.extractText())])
        self.total_token+=total
      print(f'In document {pdf_directory} a total of {pdf.numPages} page(s) was identified in the PDF and  {self.total_token} token(s) processed.')
      print()
      self.total_corpus_token+= self.total_token

      return self.sentence_token

  def lower_text(self,text):
      """
      removing spaces at the begining and end of string 
      and puts the text into small letters.

      input: text
      output: small case text
      """
      return  text.strip().lower()

  def get_sentences(self,text): 
      """
      Sentence tokenization or splitting function, breaks a bunch of 
      text into a set of sentence based on the dot (.) punctuation
      mark.

      input: text
      output: tokenized sentence
      """
      return sent_tokenize(self.lower_text(text),language='french')    

  def clean_text(self,text):
      """
      This function involves any cleaning process
      to be done on the text before it goes for continues
      preprocesing. This function takes no parameter.

      input: None
      output:splitted sentences based in part of speech tagging.
        """
      return self.get_sentences(text)

  def bind_num(self,matchobj):
      """
      Function to bind French numerical numbers

      input: regular expression object
      output: binded numerical number 
      """
      return ''.join(matchobj.group(0).split(' '))

  def train_corpus(self,corpus,save_directory,word2vec=False):
    """
    Trains pretrain language model either word2vec or fastext.
    The best models simple to implement pretrained models.

    input:
      corpus: set of tokenized words in sentence.
      save_directory: directory where the model will be saved.
      word2vec: boolean to determine whether to run the word2vec or fasttext model.

    output: None
    """
    if word2vec:
      start = time.time()
      model = Word2Vec(vector_size=100, window=5, min_count=1)
      model.build_vocab(corpus)
      model.train(corpus,total_examples=model.corpus_count, epochs=model.epochs)
      model.save(f'{save_directory}/word2vec{datetime.now()}.model')
      end = time.time()
      print(f'The model compeleted training in {end-start} seconds')
    else:
      model = FastText(vector_size=100, min_count=1)
      model.build_vocab(corpus_file=corpus)
      model.train(
          corpus_file=corpus, epochs=model.epochs, total_examples=model.corpus_count, total_words=model.corpus_total_words)
      model.save(f'{save_directory}/fasttext{datetime.now()}.model')

  def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
      '''
      Input: 
          source: a string corresponding to the string you are starting with
          target: a string corresponding to the string you want to end with
          ins_cost: an integer setting the insert cost
          del_cost: an integer setting the delete cost
          rep_cost: an integer setting the replace cost
      Output:
          D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
          med: the minimum edit distance (med) required to convert the source string to the target
      '''
      # use deletion and insert cost as  1
      m = len(source) 
      n = len(target) 
      #initialize cost matrix with zeros and dimensions (m+1,n+1) 
      D = np.zeros((m+1, n+1), dtype=int) 
      for row in range(0,m): # Replace None with the proper range
          D[row,0] = 0
      for col in range(0,n): # Replace None with the proper range
          D[0,col] = 0        
      for row in range(1,m):      
          for col in range(1,n):           
              r_cost = 0
              if None: # Replace None with a proper comparison
                  # Update the replacement cost to 0 if source and target are the same
                  r_cost = None                
              # Update the cost at row, col based on previous entries in the cost matrix
              # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
              D[row,col] = None
              
      # Set the minimum edit distance with the cost found at row m, column n 
      med = None
      return D, med

  def MED(sent_01, sent_02):
      #TODO: Learn how to implement the minimum edit score from by yourself
      """
      Minimum edit distance for two sentence
      Note: Later go back and learn about it Or you can even implement it late

      input:
        - sentence 1
        - sentence 2
      
      output: minum edit score to move from sentence 1 to 2
      """
      n = len(sent_01)
      m = len(sent_02)

      matrix = [[i+j for j in range(m+1)] for i in range(n+1)]

      for i in range(1, n+1):
          for j in range(1, m+1):
              if sent_01[i-1] == sent_02[j-1]:
                  d = 0
              else:
                  d = 1

              matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
              print(matrix[i][j])

      distance_score = matrix[n][m]
    
      return distance_score

  def preprocess_text(self,texts):
      """
      This function entails further preprocessing operations
      done on the text resulting in a set of tokens for each
      splitted sentence from text using spacy part of speech
      french tagging.

      input: list of sentence tokens
      output: list of tokens associated to each sentence
      """
      ##TODO: Revise the preprocess_text function for bugs fix-ups

      total = 0
      if type(texts) == str:
        print()
        print('ProcessErrror: Your text should be in a list !!!!')
      
      else: 
        try:
          count=1
          for text in texts:
            sentences = self.clean_text(text) ##Parser
            ##Lexical analyser and symmbol table creation per sentence
            for sentence in sentences:
              sentence = re.sub(",", ' ', sentence)
              sentence = re.sub("[0-9]+\s*.[0-9]+\s*.[0]+", self.bind_num, sentence)
              sentence = re.sub(r"http\S+", "", sentence)
              sentence = re.sub("[A-Za-z0-9]*@[A-Za-z0-9]*.[A-Za-z]*", '', sentence)
              tokens = word_tokenize(sentence, language='french')
              tokens = [re.sub("[a-z]+[',’,']",'', token) for token in tokens] 
              tokens = [token for token in tokens if token != '\n'] ##Regular expression could also solve the problem
              tokens = [token for token in tokens if token not in ['*','.',',','«','(',')','»',"l'",'-',';','[',']','—',':','…','?','–','...','!','’',"'",'•','/','➢','&','|','=']] 
              tokens = [re.sub("[.,•]",'', token) for token in tokens]

              if '\n ' in tokens:
                  tokens.remove('\n ') ##Remove empty space symbols
              if len(tokens) != 0:
                self.sentence_token.append(tokens)
                total += len(tokens)
            count+=1 ##Count the number of available text
            print()
            print(f'A total number of {total} token(s) has been processed')
        except TypeError:
          print()
          print('You used an innapproriate type of data')
        
        return self.sentence_token, total

class recognition:

   def __init__(self):
      stanza.download('fr') 
      self.nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse') ##Treebank parser
      self.token_to_complex = list()
      self.sentence_verbs = list()
      self.model = 0
      self.complex_words = 0
      self.lemmatizer = FrenchLefffLemmatizer()
      ##Part of speech tagging pretrained model
      tokenizer = AutoTokenizer.from_pretrained("gilf/french-postag-model")
      model = AutoModelForTokenClassification.from_pretrained("gilf/french-postag-model")
      self.nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
      self.lexique = pd.read_table('http://www.lexique.org/databases/Lexique383/Lexique383.tsv')
      self.lexique = self.lexique.groupby('ortho').sum() ##Grouping words to remove the obstacle of grammatical category
      self.pca = PCA(n_components=1)

   def classifier1(self,model_path,tokens):
            """
            This contains the preprocessing steps and model prediction with the use
            of the trained decision tree classifie based on principal component 
            analysis.

            input: path to the trained model, tokens
            output: predicted complex word from sentence
            """
            # TODO: Try to implement the log feature engineering strategy from coursera.
            # TODO: Fix problem ValueError: Found array with 0 sample(s) (shape=(0, 19)) while a minimum of 1 is required.

            ##Preprocessing features of each token
            valid_tokens = []
            ratio = []
            for token in tokens:
              if token in self.lexique.index.values:
                valid_tokens.append(token)
                # ratio.append(round(len(valid_tokens)/len(tokens),2))
            if len(valid_tokens) != 0:
              token_features = self.lexique[self.lexique.index.isin(valid_tokens)]
              token_features_num = token_features.select_dtypes(['int64','float64'])
              token_features_num = token_features_num.replace(-np.inf,0)
              input = self.pca.fit_transform(token_features_num)          

              ##load the model from disk
              loaded_model = pickle.load(open(model_path, 'rb'))
              result = loaded_model.predict(input)

              ##Getting the list of complex words in the tokenized sentence
              token_features_new = token_features.copy()
              token_features_new['class'] = result
              token_features_new['class'] = token_features_new['class'].replace(to_replace=[1,0], value=['simple', 'complex'])
              self.complex_words = token_features_new[token_features_new['class'] == 'complex'].index.to_list()

            return self.complex_words


   def complex_word_recognition(self,sentence_list,classifier1,model,word2vec=True):
      """
      This function permits the extraction of complex words in 
      a sentence with the use of a classification model.

      input: tokenized set of sentences
      output: tokenized sentences with their associated complex words
      """
      #TODO: Fix the problem of RuntimeWarning: invalid value encountered in true_divide explained_variance_ = (S ** 2) / (n_samples - 1)
      #TODO: Fix the problem UserWarning: `grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to `aggregation_strategy="simple"` instead.grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to"
      #TODO: Update perfarmances by playing with validation test1 and test2 and also if possible play with another POS tagging model
      self.token_to_complex = list()
      final = []
      result = []
      lemme = ''
      complex_word_pos_dict = {}
      if type(sentence_list[0]) == str:
        print('TypeError:Your input value should be in a form of a list try again') ##Check data validity
      else:

        not_found = []

        if word2vec:
          self.model = Word2Vec.load(model) ## load word2vec model
        else:
          self.model = FastText.load(model)

        for sentence in sentence_list: ## Get each sentence in the sentence list
            for word in sentence: ## Get each word in each sentence
                if word not in self.model.wv.index_to_key:
                  not_found.append(word) ## If sentence not found in vocabulary update rhe vocabulary
            if len(not_found) !=0:
              self.model.build_vocab([not_found], update=True)
              self.model.train([not_found],total_examples=self.model.corpus_count, epochs=self.model.epochs) #Then retrain the model
            
            complex_words = self.classifier1(classifier1,sentence)
            self.complex_words.remove(self.model.wv.doesnt_match(complex_words)) 
            for word in self.complex_words:
                  cos_sim_avg = np.average(self.model.wv.cosine_similarities(self.model.wv[word],self.model.wv[sentence])) \
                  * self.lexique[self.lexique.index.isin([word])]['freqfilms2'].values ## Compute cosine similarity of each word with words in the vocabulary giving a specific value to differentiate betwwen complex and simple words.
                  
                  if cos_sim_avg < 1:
                    final.append(word)
            
            ###obtaining the tag of the complex word
            for item in self.nlp_token_class(' '.join(sentence)):
              for complex_word in final:
                if item['word'] == complex_word:
                  if len(self.lemmatizer.lemmatize(complex_word,'all')) > 0:
                    if item['entity_group'] == 'V' or  item['entity_group'] == 'VIMP' or item['entity_group'] == 'VINF'or item['entity_group'] == 'VPP'or item['entity_group']=='VPR':
                      for i in self.lemmatizer.lemmatize(complex_word,'all'):
                          if i[1] == 'v':
                            lemme = i[0]                    
                            complex_word_pos_dict[complex_word] = [lemme,'VER']

                    elif item['entity_group'] == 'NPP' or item['entity_group'] == 'NC':
                      for i in self.lemmatizer.lemmatize(complex_word,'all'):
                            if i[1] == 'nc':
                              lemme = i[0]
                              complex_word_pos_dict[complex_word] = [lemme,'NC']

                    elif item['entity_group'] ==  'ADJWH' or item['entity_group'] ==  'ADJ':               
                      for i in self.lemmatizer.lemmatize(complex_word,'all'):
                            if i[1] == 'adj':
                              lemme = i[0]
                              complex_word_pos_dict[complex_word] = [lemme,'ADJ']

                    elif item['entity_group'] ==  'ADVWH' or item['entity_group'] ==  'ADV':
                      for i in self.lemmatizer.lemmatize(complex_word,'all'):
                            if i[1] == 'adv':
                              lemme = i[0]
                              complex_word_pos_dict[complex_word] = [lemme,'ADV']

                    else:
                      if item['entity_group'] == 'U' or item['entity_group'] == 'CS':
                        final.remove(complex_word)
                      # else:
                      #   print('Lemma: ',self.lemmatizer.lemmatize(complex_word,'all'))
                      #   print(item['entity_group'])
                      #   print(complex_word)
                      #   for i in self.lemmatizer.lemmatize(complex_word,'all'):
                      #           print('check: ',i[1])
                      #           lemme = i[0]                      
                      #           complex_word_pos_dict[complex_word] = [lemme,item['entity_group']]
                      #           print('I did the else case')
                      #           print()
                  
                  else:
                      final.remove(complex_word)

            result.append([' '.join(sentence),complex_word_pos_dict])

            ##Updated storage
            final = []
            complex_word_pos_dict = {}
      print()
      print('Done processsing !!')
      return  result


   def tense_recognition(self,words_sentence):
      """
      Gathers the verbs in a sentence and their associated
      genre, person number and position in text.

      input:
        - sentences: list of sentences
      
      output: 
        - sentence_verb_tense: list of sentences with their associated verbs,
          genre, person number and position in text.
      """
      sentence_verb_tense = []
      for sentence in words_sentence:       
        post,tense,person = '','',''
        verb_tense = []
        doc = self.nlp(' '.join(sentence))
        for word  in doc.sentences[0].words:
            if word.upos =='VERB':
              if word.feats is not None:
                  if len(word.feats.split('|')) > 1:      
                    for content in word.feats.split('|'):
                      if 'Tense' in content:
                        tense = content.split('=')[1]
                      elif 'Person' in  content:
                          person = content.split('=')[1] 
                    post = word.feats.split('|')[1].split('=')[1] 
                  verb_tense.append((word.id,word.text,post,person,tense))
        sentence_verb_tense.append([' '.join(sentence), verb_tense])
      return sentence_verb_tense

