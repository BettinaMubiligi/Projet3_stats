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
