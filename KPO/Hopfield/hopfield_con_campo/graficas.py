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

h=np.zeros(4)

h[0]=-1
h[1]=1
h[2]=1
h[3]=-1


patron=np.array([1.0,-1.0,-1.0,1.0],dtype=np.float64)


name="Hopfield_campo_externo_"

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


    
    
    J=np.zeros((n_oscillators,n_oscillators))


    for i in range(n_oscillators):
        for j in range(n_oscillators):
            J[i,j]=patron[i]*patron[j]/n_oscillators
    
    
#     plt.figure(rng.randint(1000),figsize=(10,6))
#     cont=0
#     s=np.ones(n_oscillators,dtype=np.float64)
#     plt.plot(tlist[1:]/K,df["Prob_conf_NL"][str(0)].loc[1:],label=str(s),color="blue",linewidth=3)
#
#     plt.title(r"Fidelity respect to the ground state of $H_{NL}$",fontsize=16)
#
#
#     plt.ylim(0,1.0)
#     plt.xlim(0.0,tlist[-1])
#     plt.ylabel("Fidelity",fontsize=16)
#     plt.xlabel("t",fontsize=16)
#     plt.tight_layout()
    #plt.savefig("graficas/ground_proyection_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".png",format="png",bbox_inches="tight")
    
    ax=plt.gca()
    ax.clear()
    
    plt.figure(rng.randint(1000),figsize=(10,6))
    cont=0
    s=np.ones(n_oscillators,dtype=np.float64)
    #plt.plot(-f(tlist)/K,fidelity[:,0],label=str(s),linestyle="--",linewidth=3)
    

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
                    aux-=0.5*J[i,j]*float(string[i]+"1")*float(string[j]+"1")
            aux+=h[i]*float(string[i]+"1")
        E.append(aux)

    E2=copy.deepcopy(E)
    E2[E2.index(np.min(E))]=100

    print(N," ",pf," ",n_oscillators,":")
    print(E)
    print("",end="\n\n\n\n")
    # print(E2)
    # print("",end="\n\n\n\n")

    plt.title("Success probability of each configuration",fontsize=16)
    plt.yticks([0.1*i for i in range(11)])
    plt.ylim(0.0,1.0)
    
    cont1=0
    cont2=0
    cont3=0
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
            
    
        if(E[j]==np.min(E)):
            plt.plot(K*tlist,df["Prob_conf"][str(j)],color="#4d5ff3",label="Global minimum" if cont1==0 else None)
            cont1=1
        elif(E[j]==np.min(E2)):
            plt.plot(K*tlist,df["Prob_conf"][str(j)],color="#11bdc7",label="Local minimum" if cont2==0 else None)
            cont2=1
        else:
            plt.plot(K*tlist,df["Prob_conf"][str(j)],color="#f09426",label="Others" if cont3==0 else None)
            cont3=1
    plt.legend(loc="lower right",fontsize=15)
    
    #plt.plot(-f(tlist)/K,fidelity[:,0]+fidelity[:,-1],linestyle="--",color="red")
    #plt.plot(-f(tlist)/K,fidelity[:,1:-1].sum(axis=1),linestyle="--",color="green")
    plt.xlim(0.0,tlist[-1])
    plt.tight_layout()
    plt.ylabel("Probability",fontsize=16)
    plt.xlabel("t",fontsize=16)
    
    plt.savefig("graficas/success_prob_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".png",format="png",bbox_inches="tight")
