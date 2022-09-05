import re
def sub_pipepline(inp):
      # def __init__(self,output):
  #   self.output=output
  

  def export(inp):

    #Extract only the sentence
    sent=inp[0]

    #Extract the complex words,lemma and pos tagging for resyf
    pairs=[]
    for k,v in inp[1].items():
        comp_wd=k
        pr=v
        # l1.append(comp_wd)
        pairs.append((comp_wd,pr))

    #extract only the complex words 
    complex_words=[]
    for i in pairs:
      comp_wd=i[0]
      # l1.append(comp_wd)
      complex_words.append(comp_wd)

    return sent,complex_words, pairs

  def camB(sentence,comp_word_list):
    # tokenise the string
    splitted = re.sub("[^\w]", " ",  sentence).split()
    new_set=[]  #empty set to add the cndidate sentences in
    
    # get comple word's position
    ind_list=[]
    for comp_word in comp_word_list:
      ind = splitted.index(comp_word)
      ind_list.append(ind)
      masked_sents=[]

      subst_set=[] #contains the candidate substitute of each complex word.
      for word in splitted:
        if splitted.index(word)==ind: #check if complex word
          masked_sent=sentence.replace(word,"<mask>") #mask the complex word
          #  masked_sents.append(masked_sent)
          output=camembert.fill_mask(masked_sent, topk= 50)
      for line in output:
            subst_set.append(line[2].strip()) #.strip() to remove white spaces at the begining of the word as cambembert generates them with white spaces
      new_set.append(subst_set)
    return new_set

  def get_only_syns(resyf_output):
    syn_list=[]
    for line in resyf_output:
      syns=list(line[1])
      syn_list.append(syns)
    o_l=[]
    for i in syn_list:
      syn_set=[]
      for elmnt in i:
        syn=elmnt[0]
        syn_set.append(syn)
      o_l.append(syn_set)

    return o_l
  # def get_only_syns(resyf_output):
  #   syn_list=[]
  #   for line in resyf_output:
  #       syns=list(line[1])
  #       syn_list.append(syns)
  #   o_l=[]
  #   for i in syn_list:
  #     syn_set=[]
  #     for elmnt in i:
  #       syn=elmnt[0]
  #       syn_set.append(syn)
  #     o_l.append(syn_set)

  #   return o_l

  # def extract_syns_rank(resyf_reslts):
  # res_syn_rankings=[]
  # for line in resyf_reslts:
  #   res_syn_rankings.append(line[1]) 
  # return res_syn_rankings

  def extract_syns_rank(resyf_reslts):
    res_syn_rankings=[]
    for line in resyf_reslts:
        res_syn_rankings.append(list(line[1]) )
    return res_syn_rankings

  def intersect(l1,l2):
    l1=list(l1) #l1= list of synonyms from resyf
    l2=list(l2) #list of synonyms from camemBERT
    intersection_list=[] #list in which to store the instersection of each word in a list

    i=0
    for l1[i] in l1:
      resyf_sub_set=set(l1[i])
      intersection=list(resyf_sub_set.intersection(l2[i]))
      if len(intersection): 
          intersection_list.append(intersection)
      else:
          intersection_list.append(l1[i][0]) #in case there is no intersection between both lists
    
      i=i+1

    return intersection_list

  def rank(l1,l2):

    #l1= output of extrect_only synonyms funtion to get only the synonyms and thier rankings from resyf results
    #l2= output of the intersection

    # 1) Rank the elements of the list according to resyf
    final_list=[]
    i=0
    for l1[i] in l1:
      j=0
      l=[]
      for l1[i][j] in l1[i]:
        if l1[i][j][0] in l2[i]:
          l.append(l1[i][j])

        j=j+1
      final_list.append(l)
      i=i+1

    # 2) Select the simplest word after the ranking
    b=[]
    i=0
    count=0
    for final_list[i] in final_list:
      j=0
      
      if final_list[i][j][0] not in b:
        b.append(final_list[i][j][0])
      else:

        b.append(final_list[i][j+count][0])
      i=i+1
      count=count+1
    return b

  def substitute(sent,comp_wd,intersections):
    d=dict(zip(comp_wd,intersections))
    s=" ".join([d.get(w,w) for w in sent.split()])
    return s.capitalize()


  def main():
    sentence,complex_words,pairs=export(inp)
    camB3_res=camB(sentence,complex_words)
    resyf_res=resf.get_all_synonyms(pairs)
    # if None in resyf_res:
    #   return sentence.capitalize()
    # else: 
    syn_rank=extract_syns_rank(resyf_res)
    only_syn=get_only_syns(resyf_res)
    intersection=intersect(only_syn,camB3_res)
    substitutions=rank(syn_rank,intersection)
    #   # return resyf_res
    return substitute(sentence,complex_words,substitutions)
    # return intersection
   
      

  return main()

