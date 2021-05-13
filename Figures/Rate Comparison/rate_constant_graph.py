# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:42:12 2021

@author: Franz Richter
"""
import numpy as np
import math
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

class CRNN:
    def __init__(self):

        self.Arrhenius = np.array([[222.4, 0, 17.3, 0],
                               [117.2, 0.15, 14.92, 0],
                               [218.0, 0.34, 36.75, 0],
                               [88.7, 0.05, 22.69, 0.45],
                               [110.5, 0.03, 14.08, 0.33],
                               [187.4, 0.04, 33.64, 0.19]])
        self.lb = 1e-9
        self.lnO2 = np.log(self.lb)
        self.R = 8.314E-3
        
    def cal_k(self, Ts):
        lnA = self.Arrhenius[:, 2]
        b = self.Arrhenius[:, 1]
        Ea = self.Arrhenius[:, 0]
        nu = self.Arrhenius[:, 3]
            
        rates = []
        for i in range(6):
            rates.append(self.get_rates(Ea[i], lnA[i], b[i], nu[i], T))

        return rates
    
    def get_rates(self, E, A, b, nu, T):
        return (lambda lnA = A, b = b, Ea = E, T = T, R = self.R, nu = nu, lnO2 = self.lnO2:
                [np.exp(nu * lnO2 + lnA + b*np.log(x) - Ea / (R * x)) for x in T])()
    
class LITR:
    def __init__(self):
        print("Class started")
        
    def cal_burnham_rates(self, T):
        As = [13, 15]
        Es = [197, 217]
        
        rates =[]
        for E in Es:
            for A in As:
                rates.append(self.cal_rates(E, A, T))
                
        
        return self.get_min_max_rate(rates, T)
        
        
        
    def cal_rates(self, E, A, T):
        rate = (lambda A = A, E=E*1000, T = T:
                [(10 ** A) * math.exp(-E/(8.314 * x)) for x in T])()
        return rate
    
    def get_min_max_rate(self, rates, T):
        min_rate = []
        max_rate = []
        for pos in range(len(T)):
            rates_at_pos = (lambda i = pos, rates = rates: [x[i] for x in rates])()
            min_rate.append(min(rates_at_pos))
            max_rate.append(max(rates_at_pos))
        return [min_rate, max_rate]
        
    def get_dauenhauer_rates(self):
        return pd.read_csv("Dauenhauerdata.csv", skiprows = 1)
    
    def cal_antal_rates(self, T):
        data = pd.read_csv("Antal.csv")
        data["rate"] = data.apply(lambda row: self.cal_rates(row["E"], 
                                                             row["log A"],
                                                             T), 
                                  axis = 1)
        return self.get_min_max_rate(data["rate"], T)
        

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
    
        # Get Temperatures
        T = [x + 273 for x in list(range(300, 600))]
        iT = [1000/x for x in T]
        
        # Open classes
        crnn = CRNN()
        literature = LITR()
        
        #CRNN Rates
        crnn_labels = [f"R{x}" for x in range(1, 7)]
        names, linestyles = getlinestyles()
        for i, rate in enumerate(crnn.cal_k(T)):
            plt.plot(iT, rate,"k", linestyle = linestyles[names[i]],
                 label = crnn_labels[i])
        
        # Burnhame Rates
        min_rate, max_rate = literature.cal_burnham_rates(T)
        plt.fill_between(iT,min_rate, max_rate, 
                     alpha = 0.1, label = "Burnham et al 2015")

        #Dauenhauer rates
        dauenhauer_rates = literature.get_dauenhauer_rates()
        plt.plot(dauenhauer_rates["X"], dauenhauer_rates["Y"], "-o",
             label = "Cell. Consumption [Krumm et al 2016]")
        plt.plot(dauenhauer_rates["X.1"], dauenhauer_rates["Y.1"], "-o",
             label = "Furans Production [Krumm et al 2016]")

        #Antal rates
        min_rate, max_rate = literature.cal_antal_rates(T)
        plt.fill_between(iT,min_rate, max_rate, 
                     alpha = 0.1, label = "Antal et al 1980 - 2002")
        
        # Additional information
        plt.yscale("log")
        plt.legend(loc = "lower left")
        plt.xlabel("$10^3$/T (K$^{-1}$)")
        plt.ylabel("Rate Constant AT$^b$exp(-E$_a$/RT)")
        plt.savefig("figure.png")
        

        
        