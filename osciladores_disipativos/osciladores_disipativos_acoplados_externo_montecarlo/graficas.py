#!/usr/bin/env python
# coding: utf-8

# In[78]:


from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.special as scs
import scipy.integrate as sci
import sys
import os
import copy


K=None
pf=None
xi=None
tmax=None
n_oscillators=None
N=None


h=np.array([-2.2,3.1,-1.1])



name="osciladores_disipativos_motecarlo_externo_n_"

for nombre in os.listdir("resultados"):

    archivo=nombre
    
    
    
    df=pd.read_csv("resultados/"+archivo,header=[0,1],skipinitialspace=True,index_col=0)
    df.dropna(inplace=True)
    
    archivo=archivo.replace(name,"").replace("osciladores_","").replace(".csv","")
    
    archivo.split("_")
    

    n_oscillators=int(archivo.split("_")[1])
    N=int(archivo.split("_")[3])
    m=int(archivo.split("_")[5])
    delta=float(archivo.split("_")[7])
    gammam=float(archivo.split("_")[9])
    gamma=float(archivo.split("_")[11])
    etaf=float(archivo.split("_")[13])
    theta=float(archivo.split("_")[15])
    n=int(archivo.split("_")[17])
    xi=float(archivo.split("_")[19])
    tau=float(archivo.split("_")[21])
    J=float(archivo.split("_")[23])

    tmax=700

    tlist=tmax*(100**np.linspace(0.0,1,list(df.index)[-1]+1)-1)/99
    rng=np.random

    s=np.ones(n_oscillators,dtype=np.float64)

    E=[]
    aux=0.0
    for j in range(2**(n_oscillators-1)):

        string=""
        b=1
        b=b<<n_oscillators
        b+=j
        sign=str(bin(b))
        for l in range(n_oscillators):
            s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])
        for l in range(n_oscillators):
            if(s[l]==1):
                string+="+"
            else:
                string+="-"

        aux=0.0
        #print(string)
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                if(i!=j):
                    aux-=0.5*float(string[i]+"1")*float(string[j]+"1")
            aux+=float(string[i]+"1")*h[i]
        E.append(aux)

    E2=copy.deepcopy(E)
    E2[E2.index(np.min(E))]=100

    plt.figure(rng.randint(1000),figsize=(10,6))
    cont=0
    s=np.ones(n_oscillators,dtype=np.float64)
    #plt.plot(-f(tlist)/K,fidelity[:,0],label=str(s),linestyle="--",linewidth=3)
    
    plt.title("Success probability of an Ising configuration",fontsize=16)
    plt.yticks([0.1*i for i in range(11)])
    plt.ylim(0.0,1.1)


    cont1=0
    cont2=0
    cont3=0


    if(True):
        for j in range(2**(n_oscillators)):

            string=""
            b=1
            b=b<<n_oscillators
            b+=j
            sign=str(bin(b))
            for l in range(n_oscillators):
                s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])

            for l in range(n_oscillators):
                if(s[l]==1):
                    string+="+"
                else:
                    string+="-"


            plt.errorbar(tlist[::4],df["Prob_conf"][str(j)][::4],yerr=df["Error"][str(j)][::4],label=r"$\left|"+string+r"\right>$",ecolor="red",elinewidth=0.5,capsize=2,linewidth=2)
    else:
        for j in range(2**n_oscillators):

            string=""
            b=1
            b=b<<n_oscillators
            b+=j
            sign=str(bin(b))
            for l in range(n_oscillators):
                s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])

            for l in range(n_oscillators):
                if(s[l]==1):
                    string+="+"
                else:
                    string+="-"


            plt.plot(tlist,df["Prob_conf"][str(j)],label=r"$\left|"+string+r"\right>$")


    plt.legend(loc="upper left",fontsize=10)
    
    plt.xlim(5*10**-1,700)
    plt.xscale("log")
    plt.tight_layout()
    plt.ylabel("Probability",fontsize=16)
    plt.xlabel(r"$t\gamma_m$",fontsize=16)
    
    plt.savefig("graficas/success_prob_n_osc_"+str(n_oscillators)+"_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta)+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+"_J_"+str(J)+".png",format="png",bbox_inches="tight")

