#!/usr/bin/env python
# coding: utf-8

# # Estudio del estado estacionario del oscilador disipativo
# 
# El sistema que se va a estudiar va a ser un oscilador que tiene un Hamiltoniano en el rotation frame:
# 
# 
# \begin{equation}
#     H_n=\Delta a^\dagger a+i\eta\left(a^n e^{in\theta}-(a^\dagger)^n e^{-in\theta}\right)
# \end{equation}
# 
# El primer término se corresponde al detuning, mientras que el segundo es un término de squezzing que va a producir que los estados coherentes se deformen pasando de ser circulares en la función de Wigner a ser más ovalados debido a que $\Delta p\neq \Delta x$. El parámetro $\eta$ es el término que nos va a dar la fuerza del squeezing. Además, el sistema presenta una simetría $\mathbb{Z}_n$, que permite hacer el cambio $a\longrightarrow ae^{-i2\pi/n}$, por lo tanto el estado fundamental presentará una degeneración de n lóbulos separados un ángulo de $2\pi/n$. Por otro lado, $m$ va a controlar el squeezing de los estados, para valores de $m$ distintos a los de $n$ se obtendrán lóbulos más alargados que otros.
# 
# Por otro lado el tamaño de los lóbulos es proporcional al cociente entre $\eta$ y $\gamma_n$, por lo que éxiste una competencia entre el término de squeezing y nonlinear damping. Esto afectará a la forma de los lóbulos.
# 
# 
# Como el Hamiltoniano está en térmninos de los operadores de creación y destrucción, habrá un conjunto infinito de autoestados de $H$, por lo que se podría pensar que la capacidad de almacenamiento de la red es infinita. Sin embargo, lo que se va a usar como memoria asosciativa son los estados metaestables del sistema por lo que el número de memorias que se pueden almacenar será igual al número de estados metaestables que se pueden introducir en la red
# 
# Sin embargo, la dinámica del sistema viene dada en un sistema disipativo con con un Lindbladiano de la forma:
# 
# \begin{equation}
#     \dot{\rho}=-i[H_n,\rho]+\gamma_1\mathcal{D}[a]\rho+\gamma_m\mathcal{D}[a^m]\rho
# \end{equation}

# La idea de esta simulación es generar cat states de forma adiabatica usando el hecho de que en el caso de la simetría fuerte $\gamma_1=0$ y estableciendo $(n,m)=(2,4)$, el estado estacionario se corresponde a un cat state. Para llevar acabo la simulación y ver que se puedan generar estos estados mediante computación adiabática lo que se va a hacer es tomar $\eta$ como parámetro de control. Este término se va a usar para hacer la amplificación de los términos no lineales del Hamiltoniano $H_n$. La función que se va a utilizar para esta dinámica es:
# 
# \begin{equation}
#     \eta(t)=\eta_f\tanh(t/\tau)
# \end{equation}
# 
# Con esta función me aseguro que para tiempos largos el valor de eta converja a aun valor constante ($\eta_f$) mientras que para tiempos intermedios se comporte como una función lineal. El parámetro $\tau$ que introduzco, tiene como único objetivo controlar como de rápido se produce la amplificación adiabática.
# 
# El cutoff del espacio de Hilbert va a ser $N=100$, $\gamma_m=1$, $\Delta=2\gamma_m$ y $\eta_f=100\gamma_m$. Por otro lado, $\tau=100$. Por otro lado, para poder realizar mediciones al igual que con el otro modelo voy a emplear $\theta=\pi/2$ esto se debe a que los cat states generados por este método los crea en el eje imaginario, mientras que en el modelo anterior son creados en el el eje real. Por lo que, dado que el valor de $\theta$ introduce una rotación de ángulo $\theta$ de los lóbulos en el espacio de Wigner, los lóbulos obtenidos ahora estarán en el eje real.

# A diferencia de casos previos, ahora se va a llevar acabo un acoplamiento entre dos osciladores para tratar de resolver el modelo de Ising. Para llevar acabo ese proceso se va a utilizar el mismo acoplamiento que con el otro modelo:
# 
# 
# \begin{equation}
#     H_I=-\xi_0\sum_{i,j}J_{ij}a_ia_j^\dagger
# \end{equation}
# 
# Por lo tanto, la dinámica del sistema va a venir dada en varios pasos:
# 
# - $\eta=0$: En esta situación:
# 
# \begin{equation}
#     H=\sum_{i=0}H_i+H_I
# \end{equation}
# 
# Con $H_i$ siendo el Hamiltoniano del oscilador armónico del oscilador $i$-iésimo.
# 
# - $\eta\neq0$: En esta situación se tendría una dinámica dada por $H_n$ para cada oscilador.
# 
# Las restricciones que se van a poner al sistemas son las mismas que en el modelo anterior. Inicialmente se va a imponer que el Hamiltoniano esté en un estado fundamental, para lograr eso se vuelve a imponer que la constante de acoplamineto de cada oscilador $\Delta$ produzca una Hamiltoniano semidefinido positivo, para que el vacumm state siga siendo el ground state.\\
# 
# Por lo tanto, iniciando desde el ground state con $\eta=0$ pretendo comprobar si para tiempos largos el sistema es capaz de resolver el modelo de Ising

# In[7]:


from qutip import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from concurrent import futures
import tracemalloc
import time as t
import multiprocessing
from multiprocessing import shared_memory
import scipy.special as scs
import scipy.optimize as sco
import scipy.integrate as sci
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.animation import FuncAnimation


print("Núcleos disponibles:", multiprocessing.cpu_count())

N=int(sys.argv[2])
n_oscillators=int(sys.argv[1])
gamma=int(sys.argv[3])
gammam=1
delta=np.zeros(n_oscillators)
#aux=float(int(sys.argv[13]))
etaf=int(sys.argv[4])
tau=int(sys.argv[5])
xi=int(sys.argv[6])
n=int(sys.argv[7])
theta=np.pi/2
m=int(sys.argv[8])
tmax=int(sys.argv[9])
rng=np.random
hilos=int(sys.argv[10])
samples=int(sys.argv[11])
bloques=int(sys.argv[12])

f=lambda t: etaf*np.tanh(t/tau)
tlist=np.linspace(0.0,tmax,samples)

tlist=tmax*(100**np.linspace(0,1,samples)-1)/199

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

for i in range(n_oscillators):
    for j in range(n_oscillators):
        delta[i]+=xi*np.abs(J[i,j])

# delta[0]=aux
# delta[1]=aux
a=[]
ad=[]


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

ls3=[("Prob_conf",i) for i in range(2**n_oscillators)]

columns=pd.MultiIndex.from_tuples(ls3)

resultados=pd.DataFrame(columns=columns,index=range(samples))


# In[8]:


H0=qzero(N**n_oscillators)
ising=qzero(N**n_oscillators)
Ht=qzero(N**n_oscillators)


rho0=Qobj(tensor([ket2dm(basis(N,0))]*n_oscillators).full())
jump=[]


for i in range(n_oscillators):
    H0+=delta[i]*ad[i]*a[i]#+ad[i]*ad[i]*a[i]*a[i]/2.0
    #Ht-=(ad[i]*ad[i]+a[i]*a[i])/2.0
    Ht+=(a[i]**n*np.exp(theta*n*1.0j)-ad[i]**n*np.exp(-theta*n*1.0j))*1.0j
    for j in range(n_oscillators):
        if(i!=j):
            H0-=xi*J[i,j]*(ad[i]*a[j])
    jump.append(np.sqrt(gamma)*a[i])
    jump.append(np.sqrt(gammam)*(a[i]**m))


# In[9]:


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
# In[10]:


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
                vect[k,col]=ket.overlap(Qobj(estado[col].full()[:,k]))

            contfila+=1

            if(contfila==N):
                contfila=0
                contfila2+=1
        
        cont+=1
    
        for i in range(len(fila)):
            #print(i)
            resultados.loc[fila[i],("Prob_conf",contcolumna)]=np.real(vect[:,i].sum())
            archivo.to_csv("Ising_osciladores_disispativos_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+".csv")
        contcolumna+=1
        


        contfila=0
        contfila2=0
        for k in range(N**n_oscillators):

            ket=tensor(Qobj(M[j][contfila2,:]),Qobj(M2[contfila,:]))
            for col in range(len(fila)):
                vect[k,col]=ket.overlap(Qobj(estado[col].full()[:,k]))

            contfila+=1

            if(contfila==N):
                contfila=0
                contfila2+=1

        cont+=1
    
        for i in range(len(fila)):
            resultados.loc[fila[i],("Prob_conf",contcolumna)]=np.real(np.real(vect[:,i].sum()))
            archivo.to_csv("Ising_osciladores_disispativos_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+".csv")
        contcolumna+=1


        
    cont2+=1
    



    for i in share:
        i.close()
    shm1.close()
    shm2.close()
    
    
    return 1


# In[11]:


try:
    ex=futures.ThreadPoolExecutor(max_workers=hilos)
    memory_allocation=np.zeros(tlist.size-1)
    trabajos=[]
    tracemalloc.start()
    t1=t.time()
    est=rho0
    contador=0
    for j in tqdm(range(int(samples/bloques))):
        
        result=mesolve([H0,[Ht,f]],est,tlist[contador*bloques:(contador+1)*bloques],c_ops=jump,options={
            "store_final_state":True,"store_states":True,"nsteps":100000000,"progress_bar":"tqdm","normalize_output":False,"atol":1e-10,"rtol":1e-10,"method":"adams"
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
    resultados.to_csv("Ising_osciladores_disispativos_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+".csv")
    
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





