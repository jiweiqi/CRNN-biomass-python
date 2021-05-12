# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:50:08 2021

@author: Franz Richter
"""
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict


def rateconstantcrnn(T):

    rate = []
    
    A, E, b = getparameters()
    
    for i in range(6):
        rates = (lambda A = A[i], E = E[i], b = b[i], T = T: 
                 [math.exp(A) * (x ** b) * math.exp(-E/(8.314*x)) for x in T])()
        rate.append(rates)
    
    return rate

def getparameters():
    A = [17.3, 14.92, 36.75, 22.69, 14.08, 33.64]
    E = [222.4, 117.2, 218, 88.7, 110.5, 187.4]
    E = [x * 1000 for x in E]
    b = [0, 0.15, 0.34, 0.05, 0.03, 0.04]
    eps = 1e-9
    count = 0
    v = [0.45, 0.33, 0.19]
    for i in range(3,6):
        A[i] = A[i] * (eps ** v[count])
    return [A, E, b]
    



def rateconstantburnham(T):
    def calculaterates(E, A):
        rate = []
    
        for energy in E:
            for factor in A:
                rates = (lambda A = factor, E = energy, T = T: 
                 [(10 ** A) * math.exp(-E/(8.414*x)) for x in T])()
                rate.append(rates)
        return rate
        
    A = [13, 15]
    E = [197000, 217000]
    rate = calculaterates(E, A)

    min_rate = []
    max_rate = []
    
    for pos in range(len(rate[0])):
        rates_at_pos = (lambda i = pos, rates = rate: [x[i] for x in rate])()
        min_rate.append(min(rates_at_pos))
        max_rate.append(max(rates_at_pos))
    
    
    return [min_rate, max_rate]
    


def rateconstantdauenhauer():
    data = pd.read_csv("Dauenhauerdata.csv", skiprows = 1)
    return data


def rateconstantantal(T):
    def caculaterates(E, A, T):
        rate = (lambda A = A, E=E*1000, T = T:
                [(10 ** A) * math.exp(-E/(8.314 * x)) for x in T])()
        return rate
        
    
    data = pd.read_csv("Antal.csv")
    data["rate"] = data.apply(lambda row: caculaterates(row["E"],
                                        row["log A"], T),
                            axis = 1)
    data.drop(labels = 27, axis = 0, inplace = True) #get rid of the outlier
    data.reset_index(drop=True, inplace=True)
    
    # Assemble mean
    ratesmatrix = []
    for i in range(len(T)):
            ratesmatrix.append((lambda xlist = data["rate"], idx = i:
                           [x[i] for x in xlist])())
    meanrate = [np.mean(x) for x in ratesmatrix]
    stdrate = [np.std(x) for x in ratesmatrix]  
    
    
    
    return [[min(x) for x in ratesmatrix],
            [max(x) for x in ratesmatrix]]

def plottherates(T, crnn_rates, burnham_rates, dauenhauer_rates, antal_rates):
    
    inversetemperature = [1000/x for x in T]
    
    
    #CRNN rates
    crnn_labels = [f"R{x}" for x in range(1, 7)]
    names, linestyles = getlinestyles()
    
    for i, rate in enumerate(crnn_rates):
        plt.plot(inversetemperature, rate,"k", linestyle = linestyles[names[i]],
                 label = crnn_labels[i])
        
     #Burnham rates
    plt.fill_between(inversetemperature,burnham_rates[0], burnham_rates[1], 
                     alpha = 0.1, label = "Burnham et al 2015")
        
    
    #Dauenhauer rates
    plt.plot(dauenhauer_rates["X"], dauenhauer_rates["Y"], "-o",
             label = "Cell. Consumption [Krumm et al 2016]")
    plt.plot(dauenhauer_rates["X.1"], dauenhauer_rates["Y.1"], "-o",
             label = "Furans Production [Krumm et al 2016]")
    
    
    #Antal
    plt.fill_between(inversetemperature, antal_rates[0], antal_rates[1], 
                     alpha = 0.1, label = "Antal et al. 1980 - 2002")
    
    plt.yscale("log")
    plt.legend(loc = "lower left")
    plt.xlabel("$10^3$/T (K$^{-1}$)")
    plt.ylabel("Rate Constant AT$^b$exp(-E$_a$/RT)")
    plt.savefig("figure.png")
    

def getlinestyles():
    linestyles = OrderedDict(
        [('solid',               (0, ())),
         ('loosely dotted',      (0, (1, 10))),
         ('dotted',              (0, (1, 5))),
         ('densely dotted',      (0, (1, 1))),
    
         ('loosely dashed',      (0, (5, 10))),
         ('dashed',              (0, (5, 5))),
         ('densely dashed',      (0, (5, 1))),
    
         ('loosely dashdotted',  (0, (3, 10, 1, 10))),
         ('dashdotted',          (0, (3, 5, 1, 5))),
         ('densely dashdotted',  (0, (3, 1, 1, 1))),
    
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    
    names = [
        "solid",
        "dotted",
        "dashed",
        "dashdotted",
        "densely dashdotdotted",
        "densely dotted"
        ]
    
    return [names, linestyles]
    


if __name__ == '__main__':

    T = list(range(300, 600)) #temperature
    T = [x + 273 for x in T]
    crnn_rates = rateconstantcrnn(T)
    burnham_rates = rateconstantburnham(T)
    dauenhauer_rates = rateconstantdauenhauer()
    antal_rates = rateconstantantal(T)  


    plottherates(T, crnn_rates, burnham_rates, dauenhauer_rates,
             antal_rates)