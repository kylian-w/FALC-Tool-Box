# -*- coding: utf-8 -*-
"""syngen.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L9NG-8KCPS2Au-mo77flmsFGk95c3duD
"""

!wget https://dl.fbaipublicfiles.com/fairseq/models/camembert-large.tar.gz
!tar -xzvf camembert-large.tar.gz

import os
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
import re

from nltk.tokenize import TreebankWordTokenizer
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
# os.chdir('/content/camembert-base') # to use camemBERT-base
os.chdir('/content/camembert-large')  # to use camemBERT-large

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

!pip install fairseq

!pip install sentencepiece

!pip install transformers

#load model in fairseq
from fairseq.models.roberta import CamembertModel
# camembert = CamembertModel.from_pretrained('/content/camembert-base/') # to use camemBERT-base
camembert = CamembertModel.from_pretrained('/content/camembert-large/')   # to use camemBERT-large

def camB(sentence,comp_word):
  # tokenise the string
  splitted = re.sub("[^\w]", " ",  sentence).split()
  # get comple word's position
  ind = splitted.index(comp_word)
  # replace the complex word by <mask> for it to be recognised by the camemBERT model
  new_set=[]  #empty set to add the cndidate sentences in

  for word in splitted:
    if splitted.index(word)==ind: #check if complex word
      masked_sent=sentence.replace(word,"<mask>") #mask the complex word
      output=camembert.fill_mask(masked_sent, topk= 10) #topk is to specify the number of substitute candidates we want to have
      for line in output:
        new_set.append(line[0]) #take the first element of each output line as they contain the needed subctitute candisate sentence
  return new_set

sent_to_subst="nous aimerions rendre l'information plus accessible"
cw="accessible"
camB(sent_to_subst,cw)

def wn_substitution(sent,comp_word):
  s=wn.synsets(comp_word,lang='fra') #find the synset of the complex word
  

  #put the synonyms of this complex word in a set
  sett=[]
  for synset in s:
    for n in synset.lemma_names(lang='fra'):
      sett.append(n)
 

  #tokenise the sentence and get the position of the complex word
  splitted = re.sub("[^\w]", " ",  sent).split()
  i = splitted.index(comp_word)

  #create a list with all the candidate sentences with substituted comples word
  sub_list=[]
  for word in sett:
    l = splitted[:i]+[word]+splitted[i+1:]
    l=" ".join(l)
    sub_list.append(l)

  return sub_list

sent_to_subst="nous aimerions rendre l'information plus accessible"
cw="accessible"
wn_substitution(sent_to_subst,cw)

