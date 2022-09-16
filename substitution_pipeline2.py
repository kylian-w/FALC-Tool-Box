import re

def sub_pipepline(inp,camembert,resf,cg,nlp1):
      # def __init__(self,output):
  #   self.output=output
  

  # def export(inp):

  #   #Extract only the sentence
  #   sent=inp[0]

  #   #Extract the complex words,lemma and pos tagging for resyf
  #   pairs=[]
  #   for k,v in inp[1].items():
  #       comp_wd=k
  #       pr=v
  #       # l1.append(comp_wd)
  #       pairs.append((comp_wd,pr))

  #   #extract only the complex words 
  #   complex_words=[]
  #   for i in pairs:
  #     comp_wd=i[0]
  #     # l1.append(comp_wd)
  #     complex_words.append(comp_wd)

  #   return sent,complex_words, pairs

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

    # extract the lemma
    pos=[]
    for elmnt in pairs:
      pos.append(elmnt[1][1])


    #extract the morph
    m=inp[2]
    morph=[]
    for elmnt in m:
      l1=[]
      for k,v in elmnt.items():
        l1.append((k,v))
      morph.append(l1)

    return sent,complex_words, pairs,pos,morph


  def morph_ext(m):
    f=[]
    for word in m:
      li=[]
      for morph in word:
        if morph[0]=="Mood" and morph[1]=='Ind':
          m=(morph[1]+'icatif').lower()
          li.append(m)
          # print('mood = ',m)


        #-----------Number------------
        elif morph[0]=="Number":
          num=morph[1]
          li.append(num)
          # print('num = ',num)

        
        #-----------person------------
        elif morph[0]=="Person":
          pers=morph[1]
          li.append(pers)
          # print('pers = ',pers)


          #-----------Tense------------

        elif morph[0]=="Tense" and morph[1]=='Pres':
          tns='présent'
          li.append(tns)
          # print('tense = ',tns)

        elif morph[0]=="Tense" and morph[1]=='Past':
          tns='passé-composé'
          li.append(tns)
          # print('tense = ',tns)

        elif morph[0]=="Tense" and morph[1]=='Fut':
          tns='futur-simple'
          li.append(tns)
          # print('tense = ',tns)Imp


        elif morph[0]=="Tense" and morph[1]=='Imp':
          tns='imparfait'
          li.append(tns)
          # print('tense = ',tns)


        #-----------VerbForm------------

        elif morph[0]=="VerbForm":
          vf=morph[1]
          li.append(vf)
          # print('VerbForm = ',vf)


        #-----------Gender------------
        elif morph[0]=="Gender":
          gd=morph[1]
          li.append(gd)
          # print('gen = ',gd)
      f.append(li)

    return f



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


  def join(syn,pos,m):
    final=[]
    i=0
    for syn[i] in syn:
        final.append((syn[i],pos[i],m[i]))
        i=i+1
    return final

#----------conjugate the verb-------------------------------------
  def conjug(cg,nlp,joint):
    try:
      
      #--------------Get the lemma form of the verb----------------
      verb=joint[0]
      doc = nlp(verb)
      for token in doc:
          v=token.lemma_

      #----------get person----------------------------------------
      conjugation = cg.conjugate(v)
      number=joint[2][1]
      person=joint[2][2]
      if number=='Sing':
        person=person-1
      else:
        person=person+2

      #----------get mood----------------------------------------
      mood=joint[2][0]

      #----------get tense----------------------------------------
      tense=joint[2][3]

      c=str(conjugation['moods'][mood][tense][person])
      
      return c.split(' ')[1]
    except:
      return '!-'+joint[0]+'-!'

#------------------Pluralise-------------------------------------------------------------

  def pluralize(word):
    for GRAMMAR_RULE in (ail_word, al_word, au_word, eil_word, eu_word, ou_word, s_word, x_word, z_word,
            default):
        plural = GRAMMAR_RULE(word)
        if plural:
            return plural



  def ail_word(word):
      if word.endswith("ail"):
          if word == "ail":
              return "aulx"
          elif word in ("bail", "corail", u"émail", "fermail", "soupirail", "travail", "vantail", "ventail", "vitrail"):
              return word[:-3] + "aux"
          return word + "s"

  def al_word(word):
      if word.endswith("al"):
          if word in (
              "bal", "carnaval", "chacal", "festival", u"récital", u"régal",
              "bancal", "fatal", "fractal", "final", "morfal", "natal", "naval",
              u"aéronaval",
              u"anténatal", u"néonatal", u"périnatal", u"postnatal", u"prénatal",
              "tonal", "atonal", "bitonal", "polytonal",
              "corral", "deal", "goal", "autogoal", "revival", "serial", "spiritual", "trial",
              "caracal", "chacal", "gavial", "gayal", "narval", "quetzal", "rorqual", "serval",
              "metical", "rial", "riyal", "ryal",
              "cantal", "emmental", "emmenthal",
              u"floréal", "germinal", "prairial",
              ):
              return word + "s"
          return word[:-2] + "aux"

  def au_word(word):
      if word.endswith("au"):
          if word in ("berimbau", "donau", "karbau", "landau", "pilau", "sarrau", "unau"):
              return word + "s"
          return word + "x"

  def eil_word(word):
      if word.endswith("eil"):
          return "vieux" if word == "vieil" else word + "s"

  def eu_word(word):
      if word.endswith("eu"):
          if word in ("bleu", u"émeu", "enfeu", "pneu", "rebeu"):
              return word + "s"
          return word + "x"

  def ou_word(word):
      if word.endswith("ou"):
          if word in ("bijou", "caillou", "chou", "genou", "hibou", "joujou", "pou"):
              return word + "x"
          return word + "s"

  def s_word(word):
      if word[-1] == "s":
          return word

  def x_word(word):
      if word[-1] == "x":
          return word

  def z_word(word):
      if word[-1] == "z":
          return word

  def default(word):
      return word + "s"



#---------------------------Singularise-----------------------------------------------------

  def singularize(word):
      for GRAMMAR_RULE in (eau_word_sing, ail_word_sing, eil_word_sing, eu_word_sing, ou_word_sing, s_word_sing, default_sing):
          singular = GRAMMAR_RULE(word)
          if singular:
              return singular

  def eau_word_sing(word):
      if word.endswith("eaux"):
          return word[:-1]

  def ail_word_sing(word):
      if word == "aulx":
          return "ail"
      if word.endswith("aux"):
          if word in ("baux", "coraux", u"émaux", "fermaux", "soupiraux", "travaux", "vantaux", "ventaux", "vitraux"):
              return word[:-3] + "ail"
          else:
              return word[:-3] + "al"

  def eil_word_sing(word):
      if word == "vieux":
          return "vieil"

  def eu_word_sing(word):
      if word.endswith("eus") or word.endswith("eux"):
          return word[:-1]

  def ou_word_sing(word):
      if word.endswith("oux"):
          if word in ("bijoux", "cailloux", "choux", "genoux", "hiboux", "joujoux", "poux"):
              return word[:-1]
          else:
              return word

  def s_word_sing(word):
      if word.endswith("s"):
          if word in (u"abcès", u"accès", "abus", "albatros", "anchois", "anglais", "autobus", "brebis", "carquois", "cas", "chas", "colis", "concours", "corps", "cours", u"cyprès", u"décès", "devis", "discours", "dos", "embarras", "engrais", "entrelacs", u"excès", "fois", "fonds", u"gâchis", "gars", "glas", "guet-apens", u"héros", "intrus", "jars", "jus", u"kermès", "lacis", "legs", "lilas", "marais", "matelas", u"mépris", "mets", "mois", "mors", "obus", "os", "palais", "paradis", "parcours", "pardessus", "pays", "plusieurs", "poids", "pois", "pouls", "printemps", "processus", u"progrès", "puits", "pus", "rabais", "radis", "recors", "recours", "refus", "relais", "remords", "remous", u"rhinocéros", "repas", "rubis", "sas", "secours", "souris", u"succès", "talus", "tapis", "taudis", "temps", "tiers", "univers", "velours", "verglas", "vernis", "virus", "accordailles", "affres", "aguets", "alentours", "ambages", "annales", "appointements", "archives", "armoiries", u"arrérages", "arrhes", "calendes", "cliques", "complies", u"condoléances", "confins", u"dépens", u"ébats", "entrailles", u"épousailles", "errements", "fiançailles", "frais", u"funérailles", "gens", "honoraires", "matines", "mœurs", u"obsèques", u"pénates", "pierreries", u"préparatifs", "relevailles", "rillettes", u"sévices", u"ténèbres", "thermes", "us", u"vêpres", "victuailles"):
              return word
          else:
              return word[:-1]

  def default_sing(word):
      return word

#---------------------------adjust-----------------------------------------------------


  def adjust(joint_list):
    li=[]
    for line in joint_list:
      if line[1]=='VER':
        li.append(conjug(cg,nlp1,line))

      elif line[1]=='NC' and line[2][1]=='Sing':
        li.append(singularize(line[0]))

      elif line[1]=='NC' and line[2][1]=='Plur':
        li.append(pluralize(line[0]))

      else:
        li.append(line[0])
    return li


#---------------------------Substitute-----------------------------------------------------

  def substitute(sent,comp_wd,intersections):
    d=dict(zip(comp_wd,intersections))
    s=" ".join([d.get(w,w) for w in sent.split()])
    return s.capitalize()


  def main():
    sentence,complex_words,pairs,pos,morph=export(inp)
    camB3_res=camB(sentence,complex_words)
    resyf_res=resf.get_all_synonyms(pairs)
    syn_rank=extract_syns_rank(resyf_res)
    only_syn=get_only_syns(resyf_res)
    intersection=intersect(only_syn,camB3_res)
    substitutions=rank(syn_rank,intersection)
    new_morph=morph_ext(morph)
    joined=join(substitutions,pos,new_morph)
    new_subs=adjust(joined)
    return substitute(sentence,complex_words,new_subs)
    # return substitutions,pos,morph
   
  return main()

def simp_text(cwi_output,camembert,resyf,cg,nlp1):
    new_li=[]
    for elmnt in cwi_output:

        try:
            new_li.append(sub_pipepline(elmnt,camembert,resyf,cg,nlp1))
        except:
            new_li.append(elmnt[0])
    

    myjoin1 = ' '.join(new_li)

    return myjoin1