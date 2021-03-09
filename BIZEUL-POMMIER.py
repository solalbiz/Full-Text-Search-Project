###Importation de la classe Tokenizer

import re

REGEX_SPECIAL_CHARACTERS = ['+', '-', '*', '|', '&', '[', ']', '(', ')', '{',
		'}', '^', '?', '.', '$', ',', ':', '=', '#', '!', '<']

class Tokenizer:
	''' Simple tokenizer, based on provided separators. The tokenizer splits
	input strings at positions where one or more separators occur. It
	returns the tokens as a list of string. The returned list of tokens does
	not contain any empty tokens ('' string).

	Example:
	* Separators: [' ', '.']
	* Input string: 'This is a test... of the tokenizer.'
	* Output tokens: ['This', 'is', 'a', 'test', 'of', 'the', 'tokenizer']
	'''


	def __init__(self, separators):
		''' Constructor for a Tokenizer object.

		The tokenizer splits input strings based on the provided
		separators.

		:param separators: Separators of the tokenizer
		:type separators: List or array of strings
		'''
		separators = ['\\'+sep if sep in REGEX_SPECIAL_CHARACTERS else sep for sep in separators]
		self._regex = '[' + ''.join(separators) + ']+'

	def tokenize(self, s):
		''' Tokenize a string

		Tokenize a string based on the separators of the tokenizer.

		:param s: The string to tokenize.
		:type s: String

		:return: List of tokens
		:rtype: List of String
		'''
		return [t for t in re.split(self._regex, s) if t != '']
    
###Importation des modules qui seront utilisés
    
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math as m
from nltk.stem.snowball import SnowballStemmer
from timeit import default_timer as timer
from scipy import spatial

###Importation des données et des requêtes

starttotal = timer()

f = open('C:/Users/bizeu/Downloads/MDATA_TP-RITexte_data/data/datasets/med_dataset.json',)

data = json.load(f) 
f.close()

f = open('C:/Users/bizeu/Downloads/MDATA_TP-RITexte_data/data/datasets/med_queries.json',)

queries = json.load(f) 
f.close()

###Importation et Tokenization de la stoplist

Separators = ["\n",'"','>',' ','/','+', '-', '*', '|', '&', '[', ']', '(', ')', '{',
		'}', '^', '?', '.', '$', ',', ':', '=', '#', '!', '<']

sep = ['\n']

f1 = open('C:/Users/bizeu/Downloads/MDATA_TP-RITexte_data/data/stoplist/stoplist-english.txt','r')
stoplist = f1.read()
f.close()

tokenizer1 = Tokenizer(sep)
stoplist = tokenizer1.tokenize(stoplist)


###Implémentation du moteur de recherche avec index linéaire

# 1. Version simple : Recherche linéaire + modèle ensembliste

# a. Calcul descripteur ensembliste d'un document avec son sac de mots 

#Sac de mots d'un document

def SAC(doc):
    
    #Tokenization et transformation en minuscule des documents
    
    tokenizer = Tokenizer(Separators)
    tokens = tokenizer.tokenize(doc)
    tokens = [x.lower() for x in tokens]
    
    #Suppression des mots incluts dans la stoplist
    
    liste = []
    
    for x in tokens:
        ajouterMot = True
        for y in stoplist:
            if x == y:
                ajouterMot = False
        if ajouterMot:
            liste.append(x)
            
    #Racinisation des documents grâce à un SnowballStemmer
    
    stems = []
    stemmer = SnowballStemmer("english")
    
    for tok in liste:
        tok = stemmer.stem(tok)
        if tok != "":
            stems.append(tok)
            
    #Création du sac de documents
    
    sac={}#Utilisation d'un dictionnaire afin de réduire les coûts de recherche d'un mot en particulier
    for word in stems:
        if word not in sac.keys():
            sac[word] = 1
        else:
            sac[word]+= 1
            
    return sac


#Création du descripteur ensembliste 

def descripteur_ens(corpus,doc):
    
    sac = SAC(doc)
    
    return np.array(list(sac.keys()))


"""
En Python, la structure la plus appropriée pour un descripteur ensembliste est le tableau car il est 
facile de faire une union sans répétition avec np.union1d() et np.intersect1d()  
"""

#b. Mesures de similarité ensembliste

#Prennent en argument le descripteur ensembliste du document et de la requête

def Dice(d,q): 
    
    inter = np.intersect1d(d,q)
    
    return 2*inter.shape[0]/(d.shape[0]*q.shape[0])

def Jaccard(d,q): 
    
    inter = np.intersect1d(d,q)
    union = np.union1d(d,q)
    
    return inter.shape[0]/union.shape[0]

def DSN(d,q): 
    return 1-Dice(d,q)


#c. Création de l'index linéaire contenant les descripteurs ensemblistes de tous les documents

def index_lin(corpus,descripteur): #corpus au format json pour garder l'id
    return [(corpus['dataset'][i]['id'],descripteur(corpus,corpus['dataset'][i]['text'])) for i in range(len(corpus['dataset']))]

#Test du temps de construction de l'index linéaire avec descripteur ensembliste

time_index_lin = timer()

INDEX = index_lin(data, descripteur_ens)

fin_index_lin = timer()
tps_index_lin = fin_index_lin-time_index_lin

print("Le temps de construction de l'index linéaire avec un descripteur ensembliste est : "+str(tps_index_lin))


#d. Recherche dans l'index


from operator import itemgetter

#Algorithme de recherche dans l'index linéaire

def rech_ind(index,query,mesure,descripteur):
    
    res = []
    q = descripteur(queries,query.lower())#calcul du descripteur de la requête
    for x in index:
        res.append((x[0],mesure(q,x[1])))#calcul de distance entre chaque document et la requête
        
    return sorted(res,key=itemgetter(1), reverse=True)#utilisation de itemgetter pour classifier les documents du plus au moins pertinent

query = queries['queries'][0]['text']

### 2.Implémentation moteur de recherche avec index inversé

#c. Descripteurs du corpus + index inversé

#Création du vocabulaire : liste de tous les mots contenus dans le dataset et les queries

def voc():
    
    union = np.array([])
    for i in range(len(data['dataset'])):
        union = np.union1d(union,descripteur_ens(data,data['dataset'][i]['text']))
    for i in range(len(queries['queries'])):
        union = np.union1d(union,descripteur_ens(queries,queries['queries'][i]['text']))
        
    return union

voc = voc()

from collections import defaultdict
        
def index_inv(corpus,descripteur): #Le corpus est au format json afin d'avoir accès à son id

    index_inv = defaultdict(list)#on initialise un dictionnaire de listes
    ind_desc = []
    
    for i in range(len(corpus['dataset'])):
        d_id = corpus['dataset'][i]['id']#on recupère l'ID
        sac = SAC(corpus['dataset'][i]['text'])
        desc = descripteur(data,corpus['dataset'][i]['text'])
        ind_desc.append([d_id,desc])
        for j in sac.keys():
            index_inv[j].append(d_id)#pour chaque mot dans le sac du doc, on ajoute une liste contenant les ID des documents qui contiennent ce mot

    return (index_inv,ind_desc)

#Test du temps de construction de l'index inversé avec descripteur ensembliste

start = timer()

INDEX2 = index_inv(data,descripteur_ens)

end = timer()
tps_index_inv = end-start

print("Le temps de construction de l'index inversé avec un descripteur ensembliste est : "+str(tps_index_inv))

#d. Recherche dans l'index inversé 

def rech_inv(Index, query, mesure,descripteur):
    
    (index_inv,index_desc) = Index
    shortlist = np.array([])
    q_sac = SAC(query)
    q_desc = descripteur(queries,query)#calcul du sac et descripteur de la requête
    
    for t in q_sac.keys():
        shortlist = np.union1d(shortlist,index_inv[t])#création de la shortlist
    shortlist = [int(x) for x in shortlist]
    
    res = []
    for d_id in shortlist:
        res.append((d_id,mesure(q_desc,index_desc[d_id-1][1])))#calcul des similarités et tri

    return sorted(res,key=itemgetter(1), reverse=True)

#Test du temps de recherche avec l'index linéaire avec descripteur ensembliste

start = timer()

rech_ind(INDEX, query,Jaccard,descripteur_ens)[:10]

end = timer()
time_lin = end-start

#Test du temps de recherche avec l'index inversé avec descripteur ensembliste

start = timer()

rech_inv(INDEX2, query,Jaccard,descripteur_ens)[:10]

end = timer()
time_inv = end-start

print("Temps d'une recherche avec un index normal : "+ str(time_lin))
print("Temps d'une recherche avec un index inversé : "+str(time_inv))

### 3. Nouvelle version TF-IDF et cosinus

# Similarité du cosinus

def cosinus(d,q): # prend en paramètres les descripteurs vetoriels

    cos = 1
    total_q = 1
    total_d = 1 
    for i in range(len(voc)):
        b = int(d[i])
        a = int(q[i])
        total_q+= a**2
        total_d+= b**2
        cos+= a*b
        
    return cos/(m.sqrt(total_d)*m.sqrt(total_q))

def cosinus_scipy(d,q): # nous utiliserons cette distance du cosinus car elle permet de réduire le temps de calcul d'un facteur 10
    return 1- spatial.distance.cosine(d,q)

# Recherche TF

def tf(t,sac):
    
    tf = 0
    if t in sac.keys():
        tf = sac[t]     #calcul du TF d'un mot à partir de son sac
        
    return tf

def desc_vect_tf(corpus, doc):
    
    res = {}
    sac = SAC(doc)
    for t in voc:   #construction d'un dictionnaire vide dont les clés sont les mots du vocabulaire
        res[t] = 0
    for i in sac.keys():
        res[i] = tf(i,sac)  #ajout des valeurs de tf dans le dictionnaire
        
    return list(res.values())   #retourne une liste comme le descripteur ensembliste


#Test du temps de construction de l'index linéaire avec descripteur TF

start = timer()

INDEX_TF = index_lin(data,desc_vect_tf)

end = timer()
time_build_tf = end-start

#Test du temps de recherche avec l'index linéaire avec descripteur TF

start = timer()

res = rech_ind(INDEX_TF, query,cosinus_scipy,desc_vect_tf)[:10]

end = timer()
time_vect = end-start

#Test du temps de construction de l'index inversé avec descripteur TF

start = timer()

INDEX_TF_inv = index_inv(data,desc_vect_tf)

end = timer()
time_build_tf_inv = end-start

#Test du temps de recherche avec l'index inversé avec descripteur TF

start = timer()

res = rech_inv(INDEX_TF_inv, query,cosinus_scipy,desc_vect_tf)[:10]

end = timer()
time_vect_inv = end-start

print("Le temps de construction de l'index linéaire avec un descripteur vectoriel tf est : "+str(time_build_tf))
print("Le temps de construction de l'index inversé avec un descripteur vectoriel tf est : "+str(time_build_tf_inv))

print("Temps d'une recherche avec un index normal : "+ str(time_vect))
print("Temps d'une recherche avec un index inversé : "+str(time_vect_inv))

# Recherche TF-IDF

def desc_total():   #renvoie une liste de dictionnaires qui sont les sac de mots des documents
    
    desc_total = []
    for i in range(len(data["dataset"])):
        sac = SAC(data["dataset"][i]["text"])
        desc_total.append(sac)
        
    return desc_total

desc_total = desc_total()

def idf(t,desc_total):  #calcule de l'idf pour un mot
    
    df = 0
    for i in range(len(desc_total)):
        if t in desc_total[i].keys():
            df+= desc_total[i][t]
            
    return m.log(len(data["dataset"])/(df+1))

def desc_vect_idf(corpus,doc):  #cf desc_vect_tf, même fonction mais en multipliant par l'IDF dans la boucle
    
    res = {}
    sac = SAC(doc)
    for t in voc:
        res[t] = 0
    for i in sac.keys():
        res[i] = tf(i,sac)*idf(i,desc_total)
        
    return list(res.values())

#Test du temps de construction de l'index linéaire avec descripteur IDF

start = timer()

INDEX_IDF = index_lin(data,desc_vect_idf)

end = timer()
time_build_tf = end-start

#Test du temps de recherche avec l'index linéaire avec descripteur IDF

start = timer()

res = rech_ind(INDEX_IDF, query,cosinus_scipy,desc_vect_idf)[:10]

end = timer()
time_vect = end-start

#Test du temps de construction de l'index inversé avec descripteur IDF

start = timer()

INDEX_IDF_inv = index_inv(data,desc_vect_idf)

end = timer()
time_build_tf_inv = end-start

#Test du temps de recherche avec l'index inversé avec descripteur IDF

start = timer()

res = rech_inv(INDEX_IDF_inv, query,cosinus_scipy,desc_vect_idf)[:10]

end = timer()
time_vect_inv = end-start

print("Le temps de construction de l'index linéaire avec un descripteur vectoriel idf est : "+str(time_build_tf))
print("Le temps de construction de l'index inversé avec un descripteur vectoriel idf est : "+str(time_build_tf_inv))

print("Temps d'une recherche avec un index normal : "+ str(time_vect))
print("Temps d'une recherche avec un index inversé : "+str(time_vect_inv))

#Performances

f = open('C:/Users/bizeu/Downloads/MDATA_TP-RITexte_data/data/datasets/med_groundtruth.json',)

ground = json.load(f) 
f.close()

def retrieved(type_recherche,index,mesure,descripteur): #construit le dictionnaire nécessaire à partir des résultats des recherches afin d'utiliser la classe Evaluator
    
    liste_totale = []
    
    for i in range(1,30):
        res = type_recherche(index, queries['queries'][i-1]['text'],mesure,descripteur)#pour chaque requête, on recupère les documents "relevant" selon notre algorithme
        liste_relevant = []
        for j in range(len(res)):
            idd,score = res[j]#on sépare l'id du score
            liste_relevant.append(idd)#on récupère l'id
        dico = {'id':i,'relevant':liste_relevant}#on construit un dictionnaire avec l'id de la requête et la liste des ids des documents relevant
        liste_totale.append(dico)
        
    relevant = {"retrieved":liste_totale}#on concatène les listes de documents relevant dans un dictionnaire
    
    return relevant
    

class Evaluator:
	''' Class for the evaluation of information retrieval systems.

	The class allows for the computation of:
	* the (recall, precision) points for a single query, after simple
	interpolation or 11-pt interpolation,
	* the average precision (AP) for a single query,
	* the averaged 11-pt interpolated (recall, precision) points for all
	queries,
	* the mean average precision (mAP) computed over all queries.

	Each Evaluator object should be build for the evaluation of a given run
	over a given dataset. Upon construction, one should provide:
	* the search results of the run over the dataset,
	* the groundtruth of the dataset.
	Search results and groundtruth should be provided as dictionaries with
	the following structure:
	{ 'groundtruth':
		[{'id':id_1, 'relevant':[rel_11, rel12, rel13...]},
		 {'id':id_2, 'relevant':[rel_21, rel22, rel23...]},
		 ...
		]
	}
	with id_i the id of a query and [rel_i1, rel_i2, rel_i3...] the list of
	relevant / retrieved documents for this query. This list must be sorted
	by estimated relevance for retrieved documents. The root element
	('groundtruth') may be different (e.g. 'run', 'retrieved'...) for the
	dictionary of search results.

	The evaluation measures and limit cases (absence of relevant documents
	or retrieved documents) are computed in the same way as in trec_eval.
	'''

	def __init__(self, retrieved, relevant):
		''' Constructor for an Evaluator object.

		Builds an Evaluator object for a run given the lists of
		relevant documents and retrieved documents for each query.
		These lists should follow the dictionary format described in
		the documentation of the class.

		:param retrieved: List of retrieved documents for each query,
		sorted by estimated relevance.
		:param relevant: List of relevant documents (groundtruth)
		for each query.
		:type retrieved: List of Dict
		:type relevant: List of Dict
		'''
		self._retrieved = self._flatten_json_qrel(retrieved, root=list(retrieved.keys())[0])
		self._relevant = self._flatten_json_qrel(relevant)


	def _flatten_json_qrel(self, json_qrel, root='groundtruth'):
		return {item['id']:item['relevant'] for item in json_qrel[root]}


	def _interpolate_11pts(self, rp_points):
		rp_11pts = []
		recalls = np.array([rp[0] for rp in rp_points])
		precisions = np.array([rp[1] for rp in rp_points])

		for recall_cutoff in np.arange(0., 1.01, .1):
			if np.count_nonzero(recalls >= recall_cutoff) > 0:
				rp_11pts.append((recall_cutoff, np.max(precisions[recalls >= recall_cutoff])))
			else:
				rp_11pts.append((recall_cutoff, 0.))
		return rp_11pts


	def _evaluate_query_pr(self, retrieved, relevant, interpolation_11pts=True):
		# if no grountruth is available
		if relevant is None or len(relevant) == 0:
			return None

		# if nothing was retrieved
		if retrieved is None or len(retrieved) == 0:
			if interpolation_11pts:
				return [(r, 0.) for r in np.arange(0., 1.01, .1)]
			else:
				return [(0.,0.)]

		# now we can work
		rp_points = {0.:(0.,0.)}
		tps = 0
		for i, retrieved_doc_id in enumerate(retrieved):
			if retrieved_doc_id in relevant:
				tps += 1
			recall = float(tps) / float(len(relevant))
			precision = float(tps) / float(i+1)
			if recall in rp_points:
				# keep best precision for given recall
				if precision > rp_points[recall][1]:
					rp_points[recall] = (recall, precision)
			else:
				rp_points[recall] = (recall, precision)

		rp_points = [rp_points[r] for r in sorted(rp_points.keys())]

		# fix P@0
		rp_points[0] = (0., rp_points[1][1])

		if interpolation_11pts:
			rp_points = self._interpolate_11pts(rp_points)

		return rp_points


	def _evaluate_query_ap(self, retrieved, relevant):
		# if no grountruth is available
		if relevant is None or len(relevant) == 0:
			return np.nan

		# if nothing was retrieved
		if retrieved is None or len(retrieved) == 0:
			return 0.

		# now we can work
		ap = 0.
		tps = 0
		for i, retrieved_doc_id in enumerate(retrieved):
			if retrieved_doc_id in relevant:
				tps += 1
				ap += float(tps) / float(i+1)

		return ap / len(relevant)


	''' Compute the interpolated (recall, precision) points for a given
	query.

	:param query_id: ID of the query to be evaluated.
	:param interpolation_11pts: if True, 11-pt interpolation is used.
	Otherwise, regular interpolation is used (Default: True).
	:type query_id: integer or string (depending on the data intially
	provided)
	:type interpolation_11pts: Bool

	:return: (recall, precision) points
	:rtype: list of (float, float) tuples
	'''
	def evaluate_query_pr_points(self, query_id, interpolation_11pts=True):
		return self._evaluate_query_pr(self._retrieved.get(query_id), self._relevant.get(query_id), interpolation_11pts)


	''' Compute the average precision (AP) for a given query.

	:param query_id: ID of the query to be evaluated.
	:type query_id: integer or string (depending on the data intially
	provided).

	:return: The AP for the query.
	:rtype: float
	'''
	def evaluate_query_ap(self, query_id, interpolation_11pts=True):
		return self._evaluate_query_ap(self._retrieved.get(query_id), self._relevant.get(query_id))


	''' Compute the 11-pt interpolated (recall, precision) points averaged
	over the queris of the run.

	:return: averaged (recall, precision) points
	:rtype: list of (float, float) tuples
	'''
	def evaluate_pr_points(self):
		precisions = []
		for i, qid in enumerate(self._relevant.keys()):
			q_pr = self._evaluate_query_pr(self._retrieved.get(qid), self._relevant[qid], interpolation_11pts=True)
			if q_pr is not None:
				precisions.append([pr[1] for pr in q_pr])
		return list(zip(np.arange(0., 1.01, .1), np.mean(np.array(precisions), axis=0)))


	''' Compute the mean average precision (mAP) over the set of queries of
	the run.

	:return: The mAP of the run.
	:rtype: float
	'''
	def evaluate_map(self):
		aps = np.array([self._evaluate_query_ap(self._retrieved.get(qid), self._relevant[qid]) for qid in self._relevant])
		return np.mean(aps[~np.isnan(aps)])

evaluator = Evaluator(retrieved(rech_ind,INDEX,Jaccard,descripteur_ens),ground)#on initialize l'Evaluator avec la fonction retrieved
points = evaluator.evaluate_pr_points()#on calcule la courbe Rappel-Précision
mapp = evaluator.evaluate_map()#on calcule le mAP

print(mapp)

x_val = [x[0] for x in points]#ce code permet de visualiser la courbe Rappel-Précision
y_val = [x[1] for x in points]

plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
plt.show()