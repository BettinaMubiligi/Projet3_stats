#Bettina Mubiligi
#Clarisse Luong
#Emma Chakma
#Groupe 2

import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

def getPrior(train) : 
    data_true = train[train["target"]==1]
    est = len(data_true)/len(train)

    moyenne = np.mean(np.array(train['target']))  #calcul de la moyenne 
    variance = np.std(np.array(train['target']))#calcul de la variance 
  
    #Calcul de l'intervalle de confiance à 95%
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

class APrioriClassifier(AbstractClassifier) :

    def __init__(self):
        pass

    def estimClass(self, attrs):
        

    


    def statsOnDF(self, df):