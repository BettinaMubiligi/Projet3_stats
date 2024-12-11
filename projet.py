#Bettina Mubiligi
#Clarisse Luong
#Emma Chakma
#Groupe 2

import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
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
    s = ss.sem(np.array(data['target']))
    i = s * ss.t.ppf((1 + 0.95) / 2., len(data)-1)
    marge_sup = moyenne+i
    marge_inf= moyenne-i
    res = {
        'estimation' : est,
        'min5pourcent' : float(marge_inf),
        'max5pourcent': float(marge_sup)
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

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

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
                dic_res[i][int(val)] = nb_target_i/nb_negatifs
            elif i==1 : 
                dic_res[i][int(val)] = nb_target_i/nb_positifs
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
        dic_res[int(val)]={}
        nb_total = len(df[df[attr]==val])
        for i in range (0,2):
            nb_target_i = len(df[(df[attr]==val) & (df['target']==i)])
            if i==0 : 
                dic_res[int(val)][i] = nb_target_i/nb_total
            elif i==1 : 
                dic_res[int(val)][i] = nb_target_i/nb_total
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
# Nous préférons le modèle de classifieur par vraisemblance (ML2DClassifier) parce que sa précision est globalement meilleure que les autres modèles
# et aussi parce que le modèle permet une estimation plus nuancée de la classe que le modèle à postériori et le modèle à priori.
######

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

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
        print (str(len(liste_nb_val))+" variable(s) : "+ str(somme) + " octets")
    elif somme>=1024 and somme<1024*1024:
        nb_ko = somme//1024
        nb_o = somme%1024
        print (str(len(liste_nb_val))+" variable(s) : "+ str(somme) + " octets = " + str(nb_ko) + "ko " + str(nb_o) + "o")
    elif somme>=1024*1024 and somme<1024*1024*1024:
        nb_mo = somme//(1024*1024)
        nb_ko = (somme%(1024*2024))//1024
        nb_o = (somme%(1024*2024))%1024
        print (str(len(liste_nb_val))+" variable(s) : "+ str(somme) + " octets = " + str(nb_mo) + "mo " + str(nb_ko) + "ko " + str(nb_o) + "o")
    else : 
        nb_go = somme//(1024*1024*1024)
        nb_mo = (somme%(1024*1024*1024))//(1024*1024)
        nb_ko = ((somme%(1024*1024*1024))%(1024*1024))//1024
        nb_o = ((somme%(1024*1024*1024))%(1024*1024))%1024
        print (str(len(liste_nb_val))+" variable(s) : "+ str(somme) + " octets = " + str(nb_go) + "go " + str(nb_mo) + "mo " + str(nb_ko) + "ko " + str(nb_o) + "o")
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
# On cherche a prouver que P(A,B,C)=P(A)*P(B|A)*P(C|B)
# On sait que A est independant de B sachant C donc on a P(A|B,C) = P(A|C)
# On part de P(A,B,C) et on a P(A,B,C)=P(A)*P(B|A)*P(C|A,B) 
# Or, puisque A est independant de B sachant C, on a donc P(C|A,B)=P(C|B)
# On a donc bien P(A,B,C)=P(A)*P(B|A)*P(C|B)
######

######
# Question 3.3.b : Complexite en independance partielle
######
# Si les variables A, B et C ont chacune 5 valeurs, la taille en memoire avec utilisation de l'independance conditionnelle est de :
# taille_totale = taille(A)+taille(B|A)+taille(C|B) = taille(A)+taille(B)*taille(A)+taille(C)*taille(B) = 5*8+5*8*5*8+5*8*5*8 = 3 240 octets

# Si les variables A, B et C ont chacune 5 valeurs, la taille en memoire sans utilisation de l'independance conditionnelle est de : 
# taille_totale = taille(A)+taille(B)*taille(C) = 5*8+5*8*5*8 = 1 640 octets
######

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

######
# 4 -  REPRESENTATION DES INDEPENDANCES CONDITIONNELLES : MODELES GRAPHIQUES
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

######
# Question 4.2: Naive Bayes
######
"""  Décomposision de la vraisemblance P(attr1, attr2, attr3, ....|target) =  P(attr1 | target ) * P(attr2 | target )*P(attr3 | target ) ... 

    Décomposision de la distribution a posteriori  
  P(target | attr1, attr2, attr3, ....) = ( P(attr1, attr2, attr3...|target) * P(target)) / P(attr1, attr2, attr3...) """
######

######
# Question 4.3 - Modèle graphique et naïve bayes
######

######
# Question 4.3.a
######

def drawNaiveBayes(d: pd.DataFrame, colonne):
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
    liste_colonnes = d.columns.tolist()
    chaine = " "
    
    for x in liste_colonnes :
        if (x == 'target'):
            break
        else :
            chaine = chaine+colonne+'->'+x+';'
    return utils.drawGraph(chaine)

######
# Question 4.3.b
######

def nbParamsNaiveBayes(d:pd.DataFrame, target, colonnes=None):
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
# Question 4.4 - Classifieur naive bayes
######

class MLNaiveBayesClassifier(APrioriClassifier) : 
    """
    Ce classifieur  utilise le maximum de vraisemblance (ML)  pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes. 
    On utilise la fonction P2D_l(self, attr) pour avoir un 
    dictionnaire associant à la valeur t un dictionnaire asssociant à la valeur a la probabilité  P(attr=a|target=t)  .
    """ 
    def __init__(self, df):
        self.df = df
        #stockage des attributs pour le calcul des probabilités
        self.attr = [col for col in df.columns if col != 'target']
         #stockage des probabilités selon la valeur de la classe target (1 ou 0)
        self.probas_target = df['target'].value_counts(normalize=True).to_dict()
        probas = {}
        #calcul des probabilités d'appartenance à une classe pour chaque attributs, stockage dans probas
        #fonction P2D_l calcule dans df la distribution de probabilité (dictionnaire) 
        for attr in self.attr:
            probas[attr] = P2D_l(self.df, attr)
        self.probas_condit = probas


    def estimProbas(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, estime les probabilités de classe 0 ou 1 en utilisant
         le maximum de vraisemblance (ML) , pour chaque attributs.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
            Le dictionnaire probas avec chaques probabilités par attributs.
        """
        
        #probabilités à 1 pour chaque classe (au départ)
        probas = {0: 1, 1: 1}
        for target in [0, 1]:
            """ Pour chaque classe et chaque attribut, la probabilité est obtenue en multipliant la probabilité conditionnelle de chaque valeur
            d'attributs  :
            P(att1 |target_0) = P(att1_valeur1| target_0)* P(att1_valeur2| target_0)* P(att1_valeur3| target_0)*... 
            P(att1 |target_1) = P(att1_valeur1| target_1)* P(att1_valeur2| target_1)* P(att1_valeur3| target_1)*...
            """
            for attr, value in attrs.items():
                if attr in self.probas_condit and value in self.probas_condit[attr][target]:
                    probas[target] *= self.probas_condit[attr][target][value]
                else:
                    probas[target] *= 1  # Une petite valeur sinon probabilité nulles au test 
        
        return probas

    def estimClass(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, calcule les probas avec estimProbas et estime la classe correspondante, selon
        la probabilité la plus élevée.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
           La classe avec la plus grande probabilité.
        """
        probas = self.estimProbas(attrs)
        #on renvoie la classe avec la plus grande probabilité (clé 0 ou 1 de probas ayant la plus grande valeur)
        return max(probas, key=probas.get)

class MAPNaiveBayesClassifier(APrioriClassifier) : 
    """
    Ce classifieur utilise le maximum a posteriori (MAP) pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    Méthode similaire celle du classifieur qui utilise le maximum de vraisemblance (ML), mais on utilise la fonction P2D_p(self, attr) pour avoir un 
    dictionnaire associant à la valeur a un dictionnaire asssociant à la valeur t la probabilité  P(target=t|attr=a).
    """ 
    def __init__(self, df):
        self.df = df
        self.attr = [col for col in df.columns if col != 'target']
        self.probas_target_p = df['target'].value_counts(normalize=True).to_dict()
        probas_p = {}
        for attr in self.attr:
            probas_p[attr] = P2D_p(self.df, attr)
        self.probas_condit = probas_p
      

    def estimProbas(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, estime les probabilités de classe 0 ou 1 en utilisant
         le maximum a posteriori (MAP) , pour chaque attributs. Noter que probas_p est un dictionnaire de la forme
        {'attributN' : {valeur_N_de_l'attribut{ proba_classe_0: ,proba_classe_1},  valeur_N+1_de_l'attribut{ proba_classe_0: ,proba_classe_1}, etc}, 'attributN+1':{..}}}}

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
            Le dictionnaire probas avec chaques probabilités par attributs.
        """
        
        probas_p = {0: 1, 1: 1}
        for target in [0, 1]:
            for attr, value in attrs.items():
                if attr in self.probas_condit:
                    if value in self.probas_condit[attr]:
                        probas_p[target] *= self.probas_condit[attr][value][target]
                    else:
                        probas_p[target] *= 1  
        return probas_p

    def estimClass(self, attrs):
        """
        A partir d'un dictionnaire d'attributs attrs, calcule les probas avec estimProbas et estime la classe correspondante, selon
        la probabilité la plus élevée.

        Parameters
        ----------
            self : Self
            attrs : Dic[str, value]
                Le dictionnaire des differents attributs qui composent le dataframe df de self
    
        Returns
        -------
           La classe avec la plus grande probabilité.
        """
        probas = self.estimProbas(attrs)
        #on renvoie la classe avec la plus grande probabilité (clé 0 ou 1 de probas ayant la plus grande valeur)
        return max(probas, key=probas.get)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

######
# 5 - FEATURE SELECTION DANS LE CADRE DU CLASSIFIER NAIVE BAYES
######

######
# Question 5.1
######

def isIndepFromTarget(df,attr,x):
    """
        Vérifie si attr est indépendant de target au seuil de x%, utilisation de la méthode chi2_contingency de scipy.stats,  qui prend en argument
        une une table de contingence du DataFrame df,  créée avec attr et target, et renvoie la valeur  chi-carré, et de la p_valeur que l'on va comparer au seuil x.
        Parameters
        ----------
           df : pandas.DataFrame
             Le dataframe contenant les données
           attr : str
             Le nom de l'attribut à tester
           x : float
             Le seuil en pourcentage
        Returns
        -------
           Booléen True ou False 
        """
    #création de la table de contigence entre les colonnes : nombre d'occurences à 2 dimensions (fréquences) pour chaque attribut par rapport à la valeur de target
    table = pd.crosstab(df[attr], df['target'])
    """ On utilise chi2_contigency pour calculer les fréquences attendues si attr et target étaient indépendantes, puis
     on récupère la p_valeur, la probabilté d'obtenir une fréquence si les variables sont indépendantes  """
    chi2, p_valeur, dof, freq_attendue = ss.chi2_contingency(table)
    #On compare la p valeur à x, pour voir si elle dépasse bien le seuil
    if (p_valeur > x ):
        return True
    return False
######


######
# Question 5.2
######
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier) : 
    """
    Ce classifieur  utilise le maximum de vraisemblance (ML)  pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes. 
    Il utilise la fonction isIndepFromTarget(df, attr, x) pour réduire le classifieur en ne gardant que les attributs indépendants de target.
    """ 
    def __init__(self, df, x):
        self.df = df
        #on modifie légèrement le cnstructeur pour ne prendre en compte que les variables dépendantes de target
        self.attr = [col for col in df.columns if col != 'target' and isIndepFromTarget(df, col, x) is False]
        self.probas_target = df['target'].value_counts(normalize=True).to_dict()
        probas = {}

        for attr in self.attr:
            probas[attr] = P2D_l(self.df, attr)
        self.probas_condit = probas


    
    def draw(self):
        """
    Dessine un graph à partir d'une d'une liste d'attributs, transformée en chaine.

    Parameters
    ----------
      self: Self
    Returns
    -------
      Image
        l'image représentant le graphe selon les dépendances des variables.
    """
        liste_colonnes = self.attr
        chaine = " "
        
        for x in liste_colonnes :
            if (x == 'target'):
                break
            else :
                chaine = chaine+'target'+'->'+x+';'
        return utils.drawGraph(chaine)


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier) : 
    """
    Ce classifieur  utilise le maximum à posteriori (MAP)  pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes. 
    Il utilise la fonction isIndepFromTarget(df, attr, x) pour réduire le classifieur en ne gardant que les attributs indépendants de target.
    """ 
    def __init__(self, df, x):
        self.df = df
        #on modifie légèrement le cnstructeur pour ne prendre en compte que les variables dépendantes de target
        self.attr = [col for col in df.columns if col != 'target' and isIndepFromTarget(df, col, x) is False]
        self.probas_target_p = df['target'].value_counts(normalize=True).to_dict()
        probas_p = {}
        for attr in self.attr:
            probas_p[attr] = P2D_p(self.df, attr)
        self.probas_condit = probas_p
    

    
    def draw(self):
        """
    Dessine un graph à partir d'une d'une liste d'attributs, transformée en chaine.

    Parameters
    ----------
      self: Self
    Returns
    -------
      Image
        l'image représentant le graphe selon les dépendances des variables.
    """
        liste_colonnes = self.attr
        chaine = " "
        
        for x in liste_colonnes :
            if (x == 'target'):
                break
            else :
                chaine = chaine+'target'+'->'+x+';'
        return utils.drawGraph(chaine)

"""----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

######
# 6 - EVALUATION DES CLASSIFIEURS
######

######
# Question 6.1 
######
######
# Le point ideal serait celui le plus haut et le plus à droite possible. Ce sera donc celui qui representera le classifieur ayant la plus grande precision et le plus grand rappel possible.
# Pour comparer les differents classifieurs dans une representation graphique, on pourrait les positionner en fonction de leur precision et de leur rappel sur un graphe.
######

######
# Question 6.2
######

def mapClassifiers(dic, df):
    """
    A partir d'un dictionnaire dic avec pour un numero identifiant un classifieur et pour valeur le classifieur en question et un dataframe df, retourne une representation graphique de la precision et du rappel de chaque classifieur.

    Parameters
    ----------
        dic : dic
            Le dictionnaire identifiant chaque classifieur avec un numéro
        df : pandas.DataFrame
            Le dataframe sur lequel on appliquera les classifieurs
    
    Returns
    -------
        Un plt.figure dans lequel chaque clé du dictionnaire dic est positionné en fonction de la precision et du rappel du classifieur qu'elle represente.
    """
    x_list = []
    y_list = []
    z_list=[]
    for key, value in dic.items():
        x_list.append(value.statsOnDF(df)["Précision"])
        y_list.append(value.statsOnDF(df)["Rappel"])
        z_list.append(key)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_list, y_list, c="r", marker="x")
    plt.xlabel("Precision du classifieur")
    plt.ylabel("Rappel du classifieur")

    for i, txt in enumerate(z_list):
        ax.text(x_list[i], y_list[i], txt)
    
    plt.show()

######
# Question 6.3 - Conclusion
######
######
# Le classifieur avec la plus grande precision est celui n°7 (Version réduite naive Bayes utilisant le maximum de vraisemblance) car c'est le point le plus a droite qu'on obtient sur les representations graphiques appliquées à la base de données de test et d'entrainement.
# Le classifieur avec le plus grand rappel est cependant celui n°1 (Version a priori) car c'est le point le plus haut qu'on obtient dans les deux représentations graphiques.
# Il faudrait donc un classifieur utilisant ces deux methodes pour obtenir des estimations plus fiables.
######
