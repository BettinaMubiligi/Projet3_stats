#Bettina Mubiligi
#Clarisse Luong
#Emma Chakma
#Groupe 2

import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import utils
import networkx as nx
######
# 1 - CLASSIFICATION A PRIORI 
######

######
# Question 1.1 : Calcul de la probabilite a priori
######

def getPrior(data) : 
    """
    A partir d'un dataframe data, retourne un dictionnaire compose de l'estimation des cas positifs dans cette base et le calcul de l'intervalle de confiance a 95% pour cette estimation

    Parameters
    ----------
        data: pandas.DataFrame
            La dataframe des donnees qu'on va utiliser pour l'estimation
    
    Returns
    -------
        Un dictionnaire avec la probabilite d'avoir des cas positifs dans la base data ainsi que l'intervalle de confiance à 95% pour l'estimation de cette probabilite
    """

    data_true = data[data["target"]==1] # Obtention de la table avec uniquement les cas posiitifs dans data
    est = len(data_true)/len(data)

    moyenne = np.mean(np.array(data['target']))  # Calcul de la moyenne 
    variance = np.std(np.array(data['target'])) # Calcul de la variance 
  
    # Calcul de l'intervalle de confiance à 95%
    s = scipy.stats.sem(np.array(data['target']))
    i = s * scipy.stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    marge_sup = moyenne+i
    marge_inf= moyenne-i
    res = {
        'estimation' : est,
        'min5pourcent' : marge_inf,
        'max5pourcent': marge_sup
    }
    return res 

######
# Question 1.2 : Programmation orientee objet dans la hierarchie des Classifier
######

class APrioriClassifier(utils.AbstractClassifier) :
    """
    Ce classifieur permet d'estimer la classe en fonction de certains attributs.
    """ 
    def __init__(self, df):
        self.df=df
    
    #####
    # Question 1.2.a
    #####

    def estimClass(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, estime la classe 0 ou 1 de l'individu ayant ces attributs attrs.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
            La classe 0 ou 1 estimee.
        """
        if (not attrs):
            return 0
        elif (attrs["cp"]==0 | attrs["exang"]==1 | attrs["oldpeak"]>5 | attrs["ca"]>0):
            return 0
        else : 
            return 1

    #####
    # Question 1.2.b : Evaluation de classifieurs
    #####
    
    def statsOnDF(self, df):
        """
        A partir d'un dataframe df, retourne un dictionnaire avec le nombre de vrais negatifs, faux negatifs, vrais positifs et vrais negatifs estimes par le classifieur estimClass
        ainsi que la precision et le rappel du classifieur.

        Parameters
        ----------
            self : Self
            df : pandas.DataFrame
                La dataframe sur laquelle on evaluera la fiabilite du classifieur.
    
        Returns
        -------
            Un dictionnaire du nombre de vrais positifs, faux positifs, vrais negatifs et faux negatifs estimes par le classifieur ainsi que la precision et le rappel du classifieur.
        """
        vp=0
        vn=0
        fp=0
        fn=0
        for i in range(len(df)):
            dic = utils.getNthDict(df,i)
            estime = self.estimClass(dic)
            vrai = dic['target']
            if estime == 1 and vrai == 1:
                vp += 1
            if estime == 1 and vrai == 0:
                fp += 1
            if estime == 0 and vrai == 0:
                vn += 1
            if estime == 0 and vrai == 1:
                fn += 1
            
        precision = vp / (vp +fp) if (vp+fp) > 0 else 0
        rappel = vp / (vp + fn) if (vp +fn) >0 else 0

        dic_res = {'VP' : vp, 'VN' : vn, 'FP' : fp, 'FN' : fn, 'Précision' : precision, 'Rappel' : rappel}
        return dic_res

"""----------------------------------------------------------------"""

######
# 2 - CLASSIFICATION PROBABILISTE A 2 DIMENSIONS
######

######
# Question 2.1 : Probabilites conditionnelles
######

######
# Question 2.1.a
######

def P2D_l(df, attr):
    """
    A partir d'un dataframe df et d'un attribut attr, retourne un dictionnaire vec la repartition selon la valeur de target des differentes valeurs possibles d'attribut attr
    Parametersa
    ----------
        df : pandas.DataFrame
            La dataframe sur laquelle on calculera la distribution des probabilites
        attr : str
            L'attribut qu'on utilisera pour calculer la distribution des donnees dans df.
    
    Returns
    -------
        Un dictionnaire de la repartition des valeurs possibles pour target (0 ou 1) selon les differents attributs possibles attr
    """
    nb_positifs = len(df[df['target']==1])
    nb_negatifs = len(df[df['target']==0])
    liste_val_attr = df[attr].unique()
    dic_res = {1:{}, 0:{}}
    for val in liste_val_attr:
        for i in range (0,2):
            nb_target_i = len(df[(df[attr]==val) & (df['target']==i)])
            if i==0 : 
                dic_res[i][val] = nb_target_i/nb_negatifs
            elif i==1 : 
                dic_res[i][val] = nb_target_i/nb_positifs
    return dic_res

######
# Question 2.1.b
######

def P2D_p(df, attr):
    """
    A partir d'un dataframe df et d'un attribut attr, retourne un dictionnaire avec la repartition des differentes valeurs possibles d'attribut attr selon la valeur de target (0 ou 1)
    Parameters
    ----------
        df : pandas.DataFrame
            La dataframe sur laquelle on calculera la distribution des probabilites
        attr : str
            L'attribut qu'on utilisera pour calculer la distribution des donnees dans df.
    
    Returns
    -------
        Un dictionnaire de la repartition des differentes valeurs possibles pour attr selon la valeur de target (0 ou 1)
    """
    liste_val_attr = df[attr].unique()
    dic_res = {}
    for val in liste_val_attr:
        dic_res[val]={}
        nb_total = len(df[df[attr]==val])
        for i in range (0,2):
            nb_target_i = len(df[(df[attr]==val) & (df['target']==i)])
            if i==0 : 
                dic_res[val][i] = nb_target_i/nb_total
            elif i==1 : 
                dic_res[val][i] = nb_target_i/nb_total
    return dic_res

######
# Question 2.2 : Classifieurs 2D par maximum de vraisemblance
######

class ML2DClassifier(APrioriClassifier): 
    """
    Ce classifieur permet d'estimer la classe d'un invididu selon la classification a 2 dimensions par maximum de vraisemblance
    """ 
    def __init__(self, df, attr):
        dic_res = P2D_l(df, attr)
        liste_pos=[]
        liste_neg=[]
        for val in df[attr].unique(): 
            liste_pos.append(dic_res[1][val])
            liste_neg.append(dic_res[0][val])
        self.data = {attr : df[attr].unique(), 'Target=1' : liste_pos, 'Target=0' : liste_neg}
        self.df_p2dl = pd.DataFrame(self.data)
    
    def estimClass(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, estime la classe 0 ou 1 de l'individu ayant ces attributs attrs.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
            La classe 0 ou 1 estimee.
        """

        # On recupere l'attribut sur lequel on souhaite estimer
        attr = self.df_p2dl.columns[0]
        val_attr = attrs[attr]

        ligne_voulue = self.df_p2dl[self.df_p2dl[attr]==val_attr]
        # On recupere la valeur de la probabilite que target=1 pour l'attribut
        val_target_1 = ligne_voulue[ligne_voulue.columns[1]].values[0]
        # On recupere la valeur de la probabilite que target=0 pour l'attribut
        val_target_0 = ligne_voulue[ligne_voulue.columns[2]].values[0]
        if val_target_0>val_target_1:
            return 0
        else :
            return 1

######
# Question 2.3 : Classifieurs 2D par maximum a posteriori
######

class MAP2DClassifier(APrioriClassifier) : 
    """
    Ce classifieur permet d'estimer la classe d'un invididu selon la classification a 2 dimensions par maximum a posteriori
    """ 
    def __init__(self, df, attr):
        dic_res = P2D_p(df, attr)
        liste_pos=[]
        liste_neg=[]
        for val in df[attr].unique(): 
            liste_pos.append(dic_res[val][1])
            liste_neg.append(dic_res[val][0])
        self.data = {attr : df[attr].unique(), 'Target=1' : liste_pos, 'Target=0' : liste_neg}
        self.df_p2dp = pd.DataFrame(self.data)
    
    def estimClass(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, estime la classe 0 ou 1 de l'individu ayant ces attributs attrs.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
            La classe 0 ou 1 estimee.
        """

        # On recupere l'attribut sur lequel on souhaite estimer
        attr = self.df_p2dp.columns[0]
        val_attr = attrs[attr]

        ligne_voulue = self.df_p2dp[self.df_p2dp[attr]==val_attr]
        # On recupere la valeur de la probabilite que target=1 pour l'attribut
        val_target_1 = ligne_voulue[ligne_voulue.columns[1]].values[0]
        # On recupere la valeur de la probabilite que target=0 pour l'attribut
        val_target_0 = ligne_voulue[ligne_voulue.columns[2]].values[0]
        if val_target_0>val_target_1:
            return 0
        else :
            return 1
    
######
# Question 2.4 : Comparaison
######
# Nous préférons .. parce que sa précision ...
# et aussi parce que ...
######

"""----------------------------------------------------------------"""

######
# 3 - COMPLEXITES
######

######
# Question 3.1 - Complexite en memoire
######

def nbParams(data, attrs=[]): 
    """
    A partir d'un dataframe df et d'une liste d'attributs attrs, retourne la taille en memoire des tables de probabilites p(target|attr1,...,attrk)

    Parameters
    ----------
        data : pandas.DataFrame
            Le dataframe sur lequel on calculera les tables de probabilites
        attrs : list
            La liste des differents attributs sur lequel on veut calculer les probabilites conditionnelles P(target|attr)
    
    Returns
    -------
        La taille en octets des valeurs en memoire
    """
    somme=8 # Taille d'un float
    if attrs==[] : 
        liste_nb_val=[]
        for attr in list(data.columns):
            liste_nb_val.append(len(data[attr].unique()))
    else : 
        liste_nb_val=[]
        for attr in attrs : 
            liste_nb_val.append(len(data[attr].unique()))

    for val in liste_nb_val:
        somme*=val
    
    if somme<1024:
        print (str(len(attrs))+" variable(s) : "+ str(somme) + " octets")
    elif somme>=1024 and somme<1024*1024:
        nb_ko = somme//1024
        nb_o = somme%1024
        print (str(len(attrs))+" variable(s) : "+ str(somme) + " octets = " + str(nb_ko) + "ko " + str(nb_o) + "o")
    elif somme>=1024*1024 and somme<1024*1024*1024:
        nb_mo = somme//(1024*1024)
        nb_ko = (somme%(1024*2024))//1024
        nb_o = (somme%(1024*2024))%1024
        print (str(len(attrs))+" variable(s) : "+ str(somme) + " octets = " + str(nb_mo) + "mo " + str(nb_ko) + "ko " + str(nb_o) + "o")
    else : 
        nb_go = somme//(1024*1024*1024)
        nb_mo = (somme%(1024*1024*1024))//(1024*1024)
        nb_ko = ((somme%(1024*1024*1024))%(1024*1024))//1024
        nb_o = ((somme%(1024*1024*1024))%(1024*1024))%1024
        print (str(len(attrs))+" variable(s) : "+ str(somme) + " octets = " + str(nb_go) + "go " + str(nb_mo) + "mo " + str(nb_ko) + "ko " + str(nb_o) + "o")
    return somme


######
# Question 3.2 - Complexite en memoire sous hypothese d'independance complete
######

def nbParamsIndep(data): 
    """
    A partir d'un dataframe df retourne la taille en memoire des tables de probabilites 

    Parameters
    ----------
        data : pandas.DataFrame
            Le dataframe sur lequel on calculera les tables de probabilites
    
    Returns
    -------
        La taille en octets des valeurs en memoire
    """
    taille_float = 8
    somme = 0
    for attr in list(data.columns):
        somme+=len(data[attr].unique())*taille_float
    print (str(len(data.columns))+" variable(s) : "+ str(somme) + " octets")
    return somme

######
# Question 3.3 : Independance conditionnelle
######

######
# Question 3.3.a : Preuve
######
# 
######

######
# Question 3.3.b : Complexite en independance partielle
######
# 
######

######
# Question 4.1: Exemple
######
""" Contruction d'un graphe orienté pour représenter une factorisation de loi jointe avec des variables indépendantes :
    On utilise la fonction utils.drawGraphHorizontal("E;D;C;B;A") où représentée par un nœud isolé sans aucune arête.
 """    
""" Contruction d'un graphe orienté pour représenter une factorisation de loi jointe avec des variables sans indépendance :
    On utilise la fonction utils.drawGraphHorizontal("A->B->C;A->D->E").
 """ 

######
# Question 4.2: Naive Bayes
######
"""  Décomposision de la vraisemblance P(attr1, attr2, attr3, ....|target) =  P(attr1 | target ) * P(attr2 | target )*P(attr3 | target ) ... 

    Décomposision de la distribution a posteriori  
  P(target | attr1, attr2, attr3, ....) = ( P(attr1, attr2, attr3...|target) * P(target)) / P(attr1, attr2, attr3...) """

######
#
######
# Question 4.3: modèle graphique et naïve bayes
######
#4.3.a
"""
    A partir d'un dataframe et du nom de la colonne qui est la classe, dessine le graphe
     du  modèle naïve bayes  où le noeud target est l'unique parent de tous les attributs.
    Parameters
    ----------
        d : pandas.DataFrame
            Le dataframe sur lequel on tracera le graphe
        colonne : la colonne indépendante conditionellement (ici, target)
    
    Returns
    -------
        Le graphe du modèle naive Bayes.
    """
def drawNaiveBayes(d: pd.DataFrame, colonne):
    liste_colonnes = d.columns.tolist()
    chaine = " "
    
    for x in liste_colonnes :
        if (x == 'target'):
            break
        else :
            chaine = chaine+colonne+'->'+x+';'
    return utils.drawGraph(chaine)

#4.3.b
"""
    A partir d'un dataframe df retourne la taille en memoire des tables de probabilites avec le modèle
    naive Bayes

    Parameters
    ----------
        d : pandas.DataFrame
            Le dataframe sur lequel on calculera les tables de probabilites
        colonne : la colonne, ici target, indépendament conditionnelle
    
    Returns
    -------
        La taille en octets des valeurs en memoire
    """
def nbParamsNaiveBayes(d:pd.DataFrame, target, colonnes=None):
    somme = len(d[target].unique())*8
    if colonnes is None:
        colonnes = d.columns.tolist()
    
    for attr in colonnes:
        if (attr != target):
            somme+= (len(d[target].unique())*8)*len(d[attr].unique())
    if somme < 1024 :
        print (str(len(colonnes))+" variable(s) : "+ str(somme) + " octets")
    else :
        print (str(len(colonnes))+" variable(s) : "+ str(somme) + " octets"+" = "+ str(somme//1024)+"ko "+str(somme %1024)+"o")
    return somme

######
