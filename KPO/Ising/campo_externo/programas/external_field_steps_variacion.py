#!/usr/bin/env python
# coding: utf-8

# # Modelo de Ising sin campo externo
# 
# Resuelvo el problema de Ising usando computación cuántica adiabática.
# 
# \begin{equation}
#     H=\sum_{i} \left(\Delta_i a^\dagger_i a_i+\frac{K}{2}(a^\dagger_i)^2a^2_i-p(t)(a^2_i+(a^\dagger_i)^2)\right)-\frac{\xi_0}{2}\sum_i \sum_j J_{ij}(a_i^\dagger a_j^\dagger+a_j^\dagger a_i^\dagger)
# \end{equation}

# In[1]:


from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.special as scs
import scipy.integrate as sci
import sys
from concurrent import futures
import tracemalloc
import time as t
import multiprocessing
from multiprocessing import shared_memory


print("Núcleos disponibles:", multiprocessing.cpu_count())

rng=np.random


K=1.0
N=int(sys.argv[1])
n_oscillators=int(sys.argv[2])
xi=float(sys.argv[3])*K
delta=np.zeros(n_oscillators)
h=np.zeros(n_oscillators)
pf=float(sys.argv[4])
tmax=float(sys.argv[5])
samples=int(sys.argv[6])
hilos=int(sys.argv[7])
bloques=int(sys.argv[8])


h=np.array([3,-2,2,-3])

J=np.zeros((n_oscillators,n_oscillators))

for i in range(n_oscillators):
    for j in range(i,n_oscillators):
        if(rng.uniform(0.0,1.0)<0.5):
            J[i,j]=1.0#+0.1*(rng.uniform(0.0,1.0)*2-1.0)
        else:
            J[i,j]=-1.0#-0.1*(rng.uniform(0.0,1.0)*2-1.0)


J=np.ones((n_oscillators,n_oscillators))

J=(J+J.T)/2

for i in range(n_oscillators):
    J[i,i]=0


a=[]
ad=[]

for i in range(n_oscillators):
    for j in range(n_oscillators):
        delta[i]+=xi*np.abs(J[i,j])

for i in range(n_oscillators):
    delta[i]=2*K#xi*(n_oscillators-1)

for i in range(n_oscillators):
    if(i==0):
        a.append(Qobj(tensor(destroy(N),qeye(N**(n_oscillators-i-1))).data_as("dia_matrix")))
        ad.append(Qobj(tensor(create(N),qeye(N**(n_oscillators-i-1))).data_as("dia_matrix")))
    elif(i==n_oscillators-1):
        a.append(Qobj(tensor(qeye(N**i),destroy(N)).data_as("dia_matrix")))
        ad.append(Qobj(tensor(qeye(N**i),create(N)).data_as("dia_matrix")))
    else:        
        a.append(Qobj(tensor(qeye(N**i),destroy(N),qeye(N**(n_oscillators-i-1))).data_as("dia_matrix")))
        ad.append(Qobj(tensor(qeye(N**i),create(N),qeye(N**(n_oscillators-i-1))).data_as("dia_matrix")))


psi0=basis(N**n_oscillators,0)

tlist=np.linspace(0.0,tmax/K,samples)

H0=qzero(N**n_oscillators)
ising=qzero(N**n_oscillators)
Ht=qzero(N**n_oscillators)
H2=qzero(N**n_oscillators)
Ht2=qzero(N**n_oscillators)
f=lambda t:-pf*t*K*K#np.tanh(3*t/100)
alpha=lambda t:np.sqrt((-f(t)-delta[0]*np.tanh(-f(t)/(delta[0])))/K)


for i in range(n_oscillators):
    H0+=delta[i]*ad[i]*a[i]+K*ad[i]*ad[i]*a[i]*a[i]/2.0
    Ht+=(ad[i]*ad[i]+a[i]*a[i])/2.0
    for j in range(n_oscillators):
        if(i!=j):
            H0-=xi*J[i,j]*(ad[i]*a[j])
    Ht2+=xi*h[i]*(ad[i]+a[i])

ls1=[("Cat_proyection_t",i) for i in range(2**n_oscillators)]
ls3=[("Prob_conf",i) for i in range(2**n_oscillators)]

columns=pd.MultiIndex.from_tuples([*ls1,*ls3])

resultados=pd.DataFrame(columns=columns,index=range(samples))






# In[2]:


integrate=lambda x,n,m:np.sqrt(2.0/np.pi)*np.exp(-2*x**2)*scs.eval_hermite(n,np.sqrt(2)*x)*scs.eval_hermite(m,np.sqrt(2)*x)/np.sqrt(2**(n+m)*scs.gamma(n+1)*scs.gamma(m+1))
        
    
M1=np.zeros((N,N))


nombres=[]

for i in range(N):
    for j in range(N):
        M1[i,j]=sci.quad(integrate,0.0,np.inf,args=(i,j))[0]


M2=np.zeros((N,N))

for i in range(N):
    for j in range(N):
        M2[i,j]=sci.quad(integrate,-np.inf,0,args=(i,j))[0]

try:
    nombres.append("Matrix"+str(rng.randint(0,1000000000000)))
    shm1 = shared_memory.SharedMemory(name=nombres[-1],create=True, size=M1.nbytes)
    nombres.append("Matrix"+str(rng.randint(0,1000000000000)))
    shm2 = shared_memory.SharedMemory(name=nombres[-1],create=True, size=M1.nbytes)

    view1 = np.ndarray(M1.size, dtype=M1.dtype, buffer=shm1.buf)
    view2 = np.ndarray(M2.size, dtype=M2.dtype, buffer=shm2.buf)


    np.copyto(view1, M1.ravel())
    np.copyto(view2, M2.ravel())


    M1=Qobj(M1)
    M2=Qobj(M2)
    proyect=None


    memory=[]
    guardado=[]

    s=np.ones(n_oscillators,dtype=np.float64)


    cont=0

    tracemalloc.start()

    matrixcont=0
    t1=t.time()
    for j in tqdm(range(2**(n_oscillators))):

        if(j%2==0):
            b=1
            b=b<<n_oscillators
            b+=cont
            sign=str(bin(b))
            for l in range(n_oscillators):
                s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])

            ls1=[]
            ls2=[]

            for l in range(n_oscillators-1):
                if(s[l]==1.0):
                    ls1.append(M1)
                else:
                    ls1.append(M2)

            proyect=tensor(*ls1).full()

            nombres.append("Matrix"+str(rng.randint(0,1000000000000)))
            memory.append(shared_memory.SharedMemory(name=nombres[-1],create=True,size=proyect.nbytes))
            guardado.append(np.ndarray(proyect.size, dtype=proyect.dtype, buffer=memory[-1].buf))
            np.copyto(guardado[-1],proyect.ravel())
            matrixcont+=1

        cont+=1

    t2=t.time()

    print("Memoria en la declaracion de los proyectores: ",tracemalloc.get_traced_memory()[0]/1024**3)
    print("Tiempo en la declaracion de los proyectores: ",t2-t1)
    # stopping the library
    tracemalloc.stop()

except FileExistsError:
    pass

print(nombres)



def medidas_bloque2(fila,estado,archivo,nombres):



    share=[]
    M=[]
    for j in range(2**(n_oscillators-1)):
        share.append(shared_memory.SharedMemory(name=nombres[2+j]))
        M.append(np.ndarray(N**(2*(n_oscillators-1)),dtype=np.complex128,buffer=share[j].buf).reshape((N**(n_oscillators-1),N**(n_oscillators-1))))

    shared1 = shared_memory.SharedMemory(name=nombres[0])
    shared2 = shared_memory.SharedMemory(name=nombres[1])

    #print(Qobj(M[0]))
    
    M1=np.ndarray(N*N,buffer=shared1.buf).reshape((N,N))
    M2=np.ndarray(N*N,buffer=shared2.buf).reshape((N,N))

    #print(Qobj(M1))
    #raise
    fidelity=np.zeros(2**n_oscillators)

    s=np.ones(n_oscillators,dtype=np.float64)
    
    
    cont=0
    cont2=0


    for q in fila:
        cont=0
        #Genero los estados cohereres \alpha y -\alpha
        phimas=coherent(N,np.sqrt(-f(tlist[q])/K))
        phimenos=coherent(N,-np.sqrt(-f(tlist[q])/K))
        
        for j in range(2**n_oscillators):

            
            
            #En funcion del signo cual será el estado cal final
            b=1
            b=b<<n_oscillators
            b+=cont
            sign=str(bin(b))
            for l in range(n_oscillators):
                s[l]=(-1)**int(sign[len(sign)-n_oscillators+l])
                    
            ls1=[]
            ls2=[]
        
            for l in range(n_oscillators):
                if(s[l]==1.0):
                    ls1.append(phimas)
                    ls2.append(phimenos)
                else:
                    ls2.append(phimas)
                    ls1.append(phimenos)
                
                
            phi=tensor(*ls1)+tensor(*ls2)
            phi=phi.unit()
            fidelity[j]=np.abs(estado[cont2].overlap(phi))**2
            cont+=1
        cont2+=1
        for i in range(2**n_oscillators):
            archivo.loc[q,("Cat_proyection_t",i)]=fidelity[i]
            archivo.to_csv("Ising_campo_externo_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".csv")
        
            
    
    
    fidelity=np.zeros(len(fila))
    
    s=np.ones(n_oscillators,dtype=np.float64)
    
    ls3=[]
    
    cont=0
    
    
    vect=np.zeros((N**n_oscillators,len(fila)),dtype=np.complex128)

    contfila=0
    contfila2=0
    contcolumna=0

    for j in range(2**(n_oscillators-1)):

        contfila=0
        contfila2=0
        for k in range(N**n_oscillators):

            ket=tensor(Qobj(M[j][contfila2,:]),Qobj(M1[contfila,:]))
            #print(ket)
            #raise
            for col in range(len(fila)):
                vect[k,col]=ket.overlap(estado[col])

            contfila+=1

            if(contfila==N):
                contfila=0
                contfila2+=1
        
        for col in range(len(fila)):
            fidelity[col]=np.real(Qobj(vect[:,col]).overlap(estado[col]))
        cont+=1
    
        for i in range(len(fila)):
            resultados.loc[fila[i],("Prob_conf",contcolumna)]=fidelity[i]
            archivo.to_csv("Ising_campo_externo_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".csv")
        contcolumna+=1
        


        contfila=0
        contfila2=0
        for k in range(N**n_oscillators):

            ket=tensor(Qobj(M[j][contfila2,:]),Qobj(M2[contfila,:]))
            for col in range(len(fila)):
                vect[k,col]=ket.overlap(estado[col])

            contfila+=1

            if(contfila==N):
                contfila=0
                contfila2+=1

        for col in range(len(fila)):
            fidelity[col]=np.real(Qobj(vect[:,col]).overlap(estado[col]))
        cont+=1
    
        for i in range(len(fila)):
            resultados.loc[fila[i],("Prob_conf",contcolumna)]=fidelity[i]
            archivo.to_csv("Ising_campo_externo_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".csv")
        contcolumna+=1


        
    cont2+=1
    



    for i in share:
        i.close()
    shm1.close()
    shm2.close()
    
    
    return 1



# # Serializacion qutip por bloques

# In[ ]:


try:
    ex=futures.ThreadPoolExecutor(max_workers=hilos)
    memory_allocation=np.zeros(tlist.size-1)
    trabajos=[]
    tracemalloc.start()
    t1=t.time()
    est=psi0
    contador=0
    for j in tqdm(range(int(samples/bloques))):
        result=sesolve([H0,[Ht,f],[Ht2,alpha]],est,tlist[contador*bloques:(contador+1)*bloques],options={
            "store_final_state":True,"store_states":True,"nsteps":10000000,"normalize_output":False,"atol":1e-8,"rtol":1e-8
        })


        est=result.final_state
        trabajos.append(ex.submit(medidas_bloque2,[i for i in range(contador*bloques,(contador+1)*bloques)],result.states,resultados,nombres))
        contador+=1

    progress_bar = tqdm(total=100)
    suma=1
    for trab in futures.as_completed(trabajos):

        progress_bar.n=suma*100/float(samples/bloques)
        progress_bar.refresh()
        suma+=trab.result()
    progress_bar.close()
    resultados.to_csv("Ising_campo_externo_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".csv")

    t2=t.time()




    # stopping the library
    print("Allocation: ",tracemalloc.get_traced_memory()[0]/1024**2)
    print("Tiempo: ",t2-t1)

    tracemalloc.stop()

    for i in memory:
        i.close()
        i.unlink()

    shm1.close()
    shm2.close()
    shm1.unlink()
    shm2.unlink()
except KeyboardInterrupt:
    for i in memory:
        i.close()
        i.unlink()

    shm1.close()
    shm2.close()
    shm1.unlink()
    shm2.unlink()

resultados.to_csv("Ising_campo_externo_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+".csv")





