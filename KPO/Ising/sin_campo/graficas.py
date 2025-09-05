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




name="Ising_sin_campo_externo_"

for nombre in os.listdir("resultados"):

    archivo=nombre
    
    
    
    df=pd.read_csv("resultados/"+archivo,header=[0,1],skipinitialspace=True,index_col=0)
    
    archivo=archivo.replace(name,"").replace("osciladores_","").replace(".csv","")
    
    archivo.split("_")
    
    
    K=float(archivo.split("_")[1])
    pf=float(archivo.split("_")[3])
    xi=float(archivo.split("_")[5])
    tmax=float(archivo.split("_")[7])
    n_oscillators=int(archivo.split("_")[9])
    N=int(archivo.split("_")[11])
    
    tlist=np.linspace(0.0,tmax,list(df.index)[-1]+1)
    
    f=lambda t:-t*K*K*pf
    rng=np.random

    plt.figure(rng.randint(1000),figsize=(10,6))
    cont=0
    s=np.ones(n_oscillators,dtype=np.float64)
    
    plt.title("Fidelity with respect to the theoretical state",fontsize=16)
    for j in range(2**(n_oscillators-1)):
    
        b=1
        b=b<<n_oscillators
        b+=j
        sign=str(bin(b))
        for l in range(n_oscillators):
            s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])
            
        string=""
        for l in range(n_oscillators):
            if(s[l]==1):
                string+="+"
            else:
                string+="-"
        plt.plot(-f(tlist)/K,df["Cat_proyection_t"][str(j)]+df["Prob_conf"][str(2**n_oscillators-j-1)],label=r"$\left |"+string+r"\right>+inv.$")
    plt.legend(loc="lower left")
    
    #plt.ylim(,1.0)
    plt.yscale("log")
    plt.xlim(0.0,-f(tlist)[np.where(-f(tlist)<=2.5,True,False)][-1])
    plt.ylabel("Fidelity",fontsize=16)
    plt.xlabel("p(t)/K",fontsize=16)
    plt.tight_layout()
    plt.savefig("graficas/poyeccion_sobre_cat_states_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".png",format="png",bbox_inches="tight")

    E=[]
    aux=0.0
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

        aux=0.0
        #print(string)
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                if(i!=j):
                    aux-=0.5*float(string[i]+"1")*float(string[j]+"1")
        E.append(aux)

    E2=copy.deepcopy(E)
    E2[E2.index(np.min(E))]=100

    plt.figure(rng.randint(1000),figsize=(10,6))
    cont=0
    s=np.ones(n_oscillators,dtype=np.float64)
    #plt.plot(-f(tlist)/K,fidelity[:,0],label=str(s),linestyle="--",linewidth=3)
    
    plt.title("Success probability of an Ising configuration",fontsize=16)
    plt.yticks([0.1*i for i in range(11)])
    plt.ylim(0.0,1.0)


    cont1=0
    cont2=0
    cont3=0
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
            
    
        plt.plot(-f(tlist)/K,df["Prob_conf"][str(j)]+df["Prob_conf"][str(2**n_oscillators-j-1)],label=r"$\left|"+string+r"\right>$ + inv.")
    #plt.plot(-f(tlist)/K,fidelity[:,0]+fidelity[:,-1],linestyle="-.",label=r"Mínimo global")
    plt.legend(loc="lower right",fontsize=10)
    
    #plt.plot(-f(tlist)/K,fidelity[:,0]+fidelity[:,-1],linestyle="--",color="red")
    #plt.plot(-f(tlist)/K,fidelity[:,1:-1].sum(axis=1),linestyle="--",color="green")
    plt.xlim(0.0,-f(tlist)[np.where(-f(tlist)<=20,True,False)][-1])
    plt.tight_layout()
    plt.ylabel("Probability",fontsize=16)
    plt.xlabel("p(t)/K",fontsize=16)
    
    plt.savefig("graficas/success_prob_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".png",format="png",bbox_inches="tight")

