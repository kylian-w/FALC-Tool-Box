# -*- coding: utf-8 -*-
"""recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17uF0F6_x6SL_Gio0xEkgqlzFQuuhqzlV
"""
##TO-DO just keep what is needed in the imports
import PyPDF2
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy.linalg import svd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument





class preprocess:
  """
  NLP preprocessing by Engineers at ISEP

  input: text
  """
  def __init__(self):
      self.sentence_token = list()
      self.texts= list()
      self.word_token = TreebankWordTokenizer()
      self.tfidf = TfidfVectorizer()
      self.simple_vocab = 0

  def extract_text_pdf(self,pdf_directory):
      """
      This function collects text from a pdf and outputs a
      processed list of the text.

      input: pdf direcotry
      output: list of preprocessed text
      """
      pdf_clean_text = list()
      pdf =  open(pdf_directory,'rb')
      pdf = PyPDF2.PdfFileReader(pdf)
      texts= list()
      print(f'A total of {pdf.numPages} page(s) was identified in the PDF')
      for i in range(pdf.numPages):
        pagepdf = pdf.getPage(i)
        self.texts.append(self.lower_text(pagepdf.extractText()))
      self.preprocess_text(self.texts)

  def lower_text(self,text):
      return  text.strip().lower()

  def get_sentences(self,text): 
    return sent_tokenize(self.lower_text(text))    

  def clean_text(self,text):
      """
      This function involves any cleaning process
      to be done on the text before it goes for continues
      preprocesing. This function takes no parameter.

      input: None
      otput:splitted sentences based in part of speech tagging.
        """
      return self.get_sentences(text)

  def cosine_sim(self,u,v):
      """
      Cosine similarity function formulation
      """
      dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
      return dist
  

  def get_similarity(value, sentence_vocab, sentence_emb_features):
      sorted_list = sorted(sentence_vocab, key = lambda word: cos_dis(sentence_emb_features[sentence_vocab.index(value),:],sentence_emb_features[sentence_vocab.index(word),:]),reverse=True)
      return sorted_list[:5]

  def tfidfvectorizer(self,corpus):
      """
      This function creates vocabulary of simple words based on 
      a set of corporas coming from Falc and simplified versions of 
      complex phrases.

      input: corpus
      output:corpus vocabulary, embedding retaining most of the information
      """
      tfidf_dim = self.tfidf.fit_transform(corpus)
      names = tfidf.get_feature_names() #Get set of vocabulary values
      cooccurencematrix = pd.DataFrame(data = tfidf_dim.toarray(), columns = names, index = names)
      U,S,V = svd(df) ##Single value decompistion for feature extraction
      emb_features = U[:,:20]

      return tfidf.vocabulary_.keys(), emb_features




  def preprocess_text(self,texts):
      """
      This function entails further preprocessing operations
      done on the text resulting in a set of tokens for each
      splitted sentence from text using spacy part of speech
      french tagging.

      input: Corpus
      output: list of tokens associated to each sentence
      """
      if type(texts) == str:
        print('ProcessErrro: Your input value should be in a form of a list try again')
      
      else: 
        try:
          count=1
          for text in texts:
            sentences = self.clean_text(text) ##Parser
            # print(f'The length of cleaned text {count} is {len(new_text)}, that of the original text {count} is {len(text)} and their ratio is {round(len(new_text)/len(text),2)}')
            # print()##add some space
            ##Lexical analyser and symmbol table creation per sentence
            for sentence in sentences:
              tokens = self.word_token.tokenize(sentence)
              tokens = [token for token in tokens if token != '\n'] ##Regular expression could also solve the problem
              tokens = [token for token in tokens if token.isalpha()] ##Removing numerical data
              if '\n ' in tokens:
                  tokens.remove('\n ') ##Remove empty space symbols
              if len(tokens) != 0:
                self.sentence_token.append(tokens)
            count+=1 ##Count the number of available text
        except TypeError:
          print('Your data shoud be found inside a list')
        
        return self.sentence_token

class recognition:

   def __init__(self):

      self.token_to_complex = list()
      self.sentence_verbs = list()
      self.model = 0
      # self.lexique = pd.read_table('http://www.lexique.org/databases/Lexique383/Lexique383.tsv')
      # self.lexique = self.lexique.groupby('ortho').sum()


   def complex_word_recognition(self,sentence_list,path_model,margin=0.10):
      """
      This function permits the extraction of complex words in 
      a sentence with the use of a classification model.

      input: tokenized set of sentences
      output: tokenized sentences with their associated complex words
      """
      if type(sentence_list[0]) == str:
        print('TypeError:Your input value should be in a form of a list try again') ##Check data validity
      else:
        result = []
        not_found = []
        self.model = Word2Vec.load(path_model)
        for sentence in sentence_list:
            for word in sentence:
                if word not in self.model.wv.index_to_key:
                  not_found.append(word)
            if len(not_found) !=0:
              self.model.build_vocab([not_found], update=True)
              self.model.train([not_found],total_examples=self.model.corpus_count, epochs=self.model.epochs)
            for word in sentence:
                  cos_sim_avg = np.average(self.model.wv.cosine_similarities(self.model.wv[word],self.model.wv[self.model.wv.index_to_key]))

                  if cos_sim_avg > margin:
                      result.append((word,cos_sim_avg))
            print()
            result = sorted(result, key = lambda x:x[1], reverse = True)[:1]
            if len(result) !=0:
              print('Complex word(s) found')
              self.token_to_complex.append([' '.join(sentence),result[0][0]])
              
            else:
              print('No complex word found')
              self.token_to_complex.append([' '.join(sentence)])

      return self.token_to_complex  


   def tense_recognition(self):
      count = 1
      for tokens in self.tokenized_sentences: 
        ##Parse tokens
        doc = self.nlp(' '.join(tokens))
        sent = list(doc.sents)[0]
        ##Visualize parsing
        print(sent._.parse_string)

        #Extract verbs in text
        verbs = list()
        exp = re.compile('[(V]* ') ##Regular expression to extract all verbs
        for i in range(0,len(tokens)):
          word = list(doc.sents)[0][i]
          print('word',word)
          if len(list(doc.sents)[0][i]._.labels)!=0:
            if 'VN' in list(doc.sents)[0][i]._.labels and 'VINF' not in list(doc.sents)[0][i]._.parse_string:
              verbs.append(word)
          elif exp.match(list(doc.sents)[0][i]._.parse_string):
            verbs.append(word)

        ##get the word lemme and its tense value
        view = self.lexique[(self.lexique['ortho'].isin([str(i) for i in verbs])) & (self.lexique['cgram'] == 'VER')]

        ##Create output in the form of a list
        for i in range(0,len(view.ortho.values)):
          self.sentence_verbs.append({'sentence_tokens':tokens,'verb':view.ortho.values[i], 'lemme':view.lemme.values[i], 'tense':view.infover.values[i]})
        print(f'Completed sentence {count} and stored')
        print()##add some space
