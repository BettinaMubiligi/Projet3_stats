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

def getPrior(train) : 
    data_true = train[train["target"]==1]
    est = len(data_true)/len(train)

    moyenne = np.mean(np.array(train['target']))  # calcul de la moyenne 
    variance = np.std(np.array(train['target'])) # calcul de la variance 
  
    # Calcul de l'intervalle de confiance à 95%
    s = scipy.stats.sem(np.array(train['target']))
    i = s * scipy.stats.t.ppf((1 + 0.95) / 2., len(train)-1)
    marge_sup = moyenne+i
    marge_inf= moyenne-i
    res = {
        'estimation' : est,
        'min5pourcent' : marge_inf,
        'max5pourcent': marge_sup
    }
    return res 

class APrioriClassifier(utils.AbstractClassifier) :
    def __init__(self, df):
        self.df=df

    def estimClass(self, attrs):
        if (not attrs):
            return 0
        elif (attrs["cp"]==0 | attrs["exang"]==1 | attrs["oldpeak"]>5 | attrs["ca"]>0):
            return 0
        else : 
            return 1
    
    def statsOnDF(self, df):
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

def P2D_l(df, attr):
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

    
def P2D_p(df, attr):
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
    
class ML2DClassifier(APrioriClassifier): 
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
        attr = self.df_p2dl.columns[0]
        val_attr = attrs[attr]

        ligne_voulue = self.df_p2dl[self.df_p2dl[attr]==val_attr]
        val_target_1 = ligne_voulue[ligne_voulue.columns[1]].values[0]
        val_target_0 = ligne_voulue[ligne_voulue.columns[2]].values[0]
        if val_target_0>val_target_1:
            return 0
        else :
            return 1

class MAP2DClassifier(APrioriClassifier) : 
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
        attr = self.df_p2dp.columns[0]
        val_attr = attrs[attr]

        ligne_voulue = self.df_p2dp[self.df_p2dp[attr]==val_attr]
        val_target_1 = ligne_voulue[ligne_voulue.columns[1]].values[0]
        val_target_0 = ligne_voulue[ligne_voulue.columns[2]].values[0]
        if val_target_0>val_target_1:
            return 0
        else :
            return 1
