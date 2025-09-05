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


contador_g=0

name="eigen_vectors_"

for nombre in os.listdir("resultados"):

    archivo=nombre
    
    
    
    df=pd.read_csv("resultados/"+archivo,header=[0,1],skipinitialspace=True,index_col=0)
    
    archivo=archivo.replace(name,"").replace("osciladores_","").replace(".csv","")
    
    archivo.split("_")
    
    try:
        K=float(archivo.split("_")[1])
        pf=float(archivo.split("_")[3])
        xi=float(archivo.split("_")[5])
        tmax=float(archivo.split("_")[7])
        n_oscillators=int(archivo.split("_")[9])
        N=int(archivo.split("_")[11])
        M=np.zeros((2,2))
        try:
            g=float(archivo.split("_")[13])

            M[0,0]=float(archivo.split("_")[15])
            M[1,0]=float(archivo.split("_")[17])
            M[0,1]=float(archivo.split("_")[19])
            M[1,1]=float(archivo.split("_")[21])
        except:
            print(archivo)
            g=contador_g#int(archivo.split("_")[13])
            contador_g+=1
            M=np.zeros((2,2))
            M[0,0]=0#int(archivo.split("_")[15])
            M[1,0]=0#int(archivo.split("_")[17])
            M[0,1]=0#int(archivo.split("_")[19])
            M[1,1]=0#int(archivo.split("_")[21])
    except ValueError:

        try:
            archivo=archivo.replace("eigen_vectors2_","").replace("osciladores_","").replace(".csv","")

            archivo.split("_")

            K=float(archivo.split("_")[1])
            pf=float(archivo.split("_")[3])
            xi=float(archivo.split("_")[5])
            tmax=float(archivo.split("_")[7])
            n_oscillators=int(archivo.split("_")[9])
            N=int(archivo.split("_")[11])
            M=np.zeros((2,2))
            try:
                g=contador_g
                contador_g+=1

                M[0,0]=float(archivo.split("_")[15])
                M[1,0]=float(archivo.split("_")[17])
                M[0,1]=float(archivo.split("_")[19])
                M[1,1]=float(archivo.split("_")[21])
            except:
                print(archivo)
                g=0#int(archivo.split("_")[13])
                M=np.zeros((2,2))
                M[0,0]=0#int(archivo.split("_")[15])
                M[1,0]=0#int(archivo.split("_")[17])
                M[0,1]=0#int(archivo.split("_")[19])
                M[1,1]=0#int(archivo.split("_")[21])
        except ValueError:
            archivo=archivo.replace("eigen_vectors3_","").replace("osciladores_","").replace(".csv","")

            archivo.split("_")

            K=float(archivo.split("_")[1])
            pf=float(archivo.split("_")[3])
            xi=float(archivo.split("_")[5])
            tmax=float(archivo.split("_")[7])
            n_oscillators=int(archivo.split("_")[9])
            N=int(archivo.split("_")[11])
            M=np.zeros((2,2))
            try:
                g=contador_g
                contador_g+=1

                M[0,0]=float(archivo.split("_")[15])
                M[1,0]=float(archivo.split("_")[17])
                M[0,1]=float(archivo.split("_")[19])
                M[1,1]=float(archivo.split("_")[21])
            except:
                print(archivo)
                g=0#int(archivo.split("_")[13])
                M=np.zeros((2,2))
                M[0,0]=0#int(archivo.split("_")[15])
                M[1,0]=0#int(archivo.split("_")[17])
                M[0,1]=0#int(archivo.split("_")[19])
                M[1,1]=0#int(archivo.split("_")[21])



    tlist=np.linspace(0.0,tmax,list(df.index)[-1]+1)
    
    f=lambda t:-t*K*K*pf
    rng=np.random

    cont=0
    s=np.ones(n_oscillators,dtype=np.float64)
    


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
    
    plt.title("Success probability of the candidate vectors",fontsize=20)
    plt.yticks([0.1*i for i in range(11)])
    plt.ylim(0.0,1.0)


    cont1=0
    cont2=0
    cont3=0
    for j in range(2**(n_oscillators)-1,-1,-1):
    
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
    #plt.plot(-f(tlist)/K,fidelity[:,0]+fidelity[:,-1],linestyle="-.",label=r"Mínimo global")

    

    if(nombre=="eigen_vectors2_K_1.0_pf_5.0_xi_0.5_tmax_500.0_n_osciladores_4_N_9_g_<function g at 0x7f05f50bd440>_M11_2.7315076747855835_M21_-2.364591581693505_M12_-2.364591581693505_M22_9.037085225968264_tau_200.csv"):
        plt.plot(tlist,df["Prob_conf"][str(2)]+df["Prob_conf"][str(2**n_oscillators-1)],label=r"Solution")

    if(nombre=="eigen_vectors2_K_1.0_pf_5.0_xi_0.5_tmax_500.0_n_osciladores_4_N_9_g_<function g at 0x7fdf8c939440>_M11_15.75651223007325_M21_0.0_M12_0.0_M22_0.30984834845916204_tau_200.csv"):
        plt.plot(tlist,df["Prob_conf"][str(2**n_oscillators-2)]+df["Prob_conf"][str(2**n_oscillators-1)]+df["Prob_conf"][str(2**n_oscillators-4)]+df["Prob_conf"][str(2**n_oscillators-3)],label=r"Solution")
    #plt.plot(-f(tlist)/K,fidelity[:,0]+fidelity[:,-1],linestyle="--",color="red")
    #plt.plot(-f(tlist)/K,fidelity[:,1:-1].sum(axis=1),linestyle="--",color="green")
     # plt.xlim(0.0,-f(tlist)[np.where(-f(tlist)<=20,True,False)][-1])
    plt.xlim(10.0,tlist[-1])
    plt.tight_layout()
    plt.ylabel("Probability",fontsize=16)
    plt.xlabel("tK",fontsize=16)
    # plt.xscale("log")
    plt.legend(loc="upper left",fontsize=12)
    
    plt.savefig("graficas/eigen_vectors_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(M[0,0])+"_M21_"+str(M[1,0])+"_M12_"+str(M[0,1])+"_M22_"+str(M[1,1])+".png",format="png",bbox_inches="tight")

