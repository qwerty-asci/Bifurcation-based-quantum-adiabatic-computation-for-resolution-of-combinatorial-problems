#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
from qutip import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from concurrent import futures
import tracemalloc
import time as t
from contextlib import redirect_stdout
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
gammam=0.001
delta=np.zeros(n_oscillators)
etaf=float(sys.argv[4])
tau=float(sys.argv[5])
xi=float(sys.argv[6])
n=int(sys.argv[7])
theta=np.pi/2
m=int(sys.argv[8])
tmax=float(sys.argv[9])
rng=np.random
hilos=int(sys.argv[10])
samples=int(sys.argv[11])
bloques=int(sys.argv[12])
numero_de_trayectorias=int(sys.argv[13])
hilos_medida=int(sys.argv[14])
aux=float(sys.argv[15])


h=np.array([-2,3,-1])
def f(t):
    return etaf*np.tanh(t/tau)
tlist=tmax*(100**np.linspace(0,1,samples)-1)/99



def func(t):
    return f(t)**(1/6)#*(1.0/(1+np.exp(-0.05*(t-100))))

J=np.ones((3,3))
#
# J[0,1]=1
# J[0,2]=-0.9
# J[1,2]=0.7




J=(J+J.T)/2

for i in range(n_oscillators):
    J[i,i]=0
    delta[i]=aux


print(J)

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
Ht2=qzero(N**n_oscillators)


psi0=basis(N**n_oscillators,0)
jump=[]


for i in range(n_oscillators):
    H0+=delta[i]*ad[i]*a[i]#+ad[i]*ad[i]*a[i]*a[i]/2.0
    #Ht-=(ad[i]*ad[i]+a[i]*a[i])/2.0
    Ht+=(a[i]**n*np.exp(theta*n*1.0j)-ad[i]**n*np.exp(-theta*n*1.0j))*1.0j
    Ht2+=h[i]*(a[i]+ad[i])*xi
    for j in range(n_oscillators):
        if(i!=j):
            H0-=xi*J[i,j]*ad[i]*a[j]
    jump.append(np.sqrt(gammam)*(a[i]**m))



# In[35]:


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


# In[36]:


def medidas_bloque2(fila,estado,nombres,nombre_fichero,id_proceso):


    share=[]
    M=[]
    
    shared_archivo=shared_memory.SharedMemory(name="F"+str(id_proceso))
    archivo=np.ndarray(samples*2**n_oscillators,buffer=shared_archivo.buf).reshape((samples,2**n_oscillators))
    
    for j in range(2**(n_oscillators-1)):
        share.append(shared_memory.SharedMemory(name=nombres[2+j]))
        M.append(np.ndarray(N**(2*(n_oscillators-1)),dtype=np.complex128,buffer=share[j].buf).reshape((N**(n_oscillators-1),N**(n_oscillators-1))))

    shared1 = shared_memory.SharedMemory(name=nombres[0])
    shared2 = shared_memory.SharedMemory(name=nombres[1])

    
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
                vect[k,col]=ket.overlap(estado[col])

            contfila+=1

            if(contfila==N):
                contfila=0
                contfila2+=1
        
        for col in range(len(fila)):
            fidelity[col]=np.real(Qobj(vect[:,col]).overlap(estado[col]))
            
        cont+=1
    
        for i in range(len(fila)):
            archivo[fila[i],contcolumna]=fidelity[i]
            np.savetxt(nombre_fichero,archivo,delimiter=",")
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
            archivo[fila[i],contcolumna]=fidelity[i]
            np.savetxt(nombre_fichero,archivo,delimiter=",")
            
        contcolumna+=1

 
    cont2+=1
    



    for i in share:
        i.close()
    shm1.close()
    shm2.close()

    shared_archivo.close()
    
    return 1


# In[37]:


def trayectoria(estado_inicial,lista_de_hamiltonianos,operadores_de_salto,bloques,tlist,id_proceso,hilos_medidas,nombres,medidas_bloque2,id_fichero,seed):

    
    nombre_fichero_resultados="osciladores_disipativos_motecarlo_externo_id_"+str(id_fichero)+"_n_osc_"+str(n_oscillators)+"_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+"_J_"+str(J[0,1])+".csv"

    ls3=[("Prob_conf",i) for i in range(2**n_oscillators)]

    columns=pd.MultiIndex.from_tuples(ls3)
    resultados=pd.DataFrame(columns=columns,index=range(samples))

    try:
        csv = shared_memory.SharedMemory(name="F"+str(id_proceso),create=True, size=np.zeros((samples,2**n_oscillators)).nbytes)
    
        contenido = np.ndarray(np.zeros((samples,2**n_oscillators)).size, dtype=np.float64, buffer=csv.buf)
        
        np.copyto(contenido, np.zeros((samples,2**n_oscillators)).ravel())
    except FileExistsError:
        csv=shared_memory.SharedMemory(name="F"+str(id_proceso))
        csv.unlink()
        csv.close()

        csv = shared_memory.SharedMemory(name="F"+str(id_proceso),create=True, size=np.zeros((samples,2**n_oscillators)).nbytes)
    
        contenido = np.ndarray(np.zeros((samples,2**n_oscillators)).size, dtype=np.float64, buffer=csv.buf)
        
        np.copyto(contenido, np.zeros((samples,2**n_oscillators)).ravel())

    
    
    ex=futures.ProcessPoolExecutor(max_workers=hilos_medidas)
    trabajos=[]
    est=estado_inicial
    contador=0
    rn=np.random.default_rng(seed)
    try:

        print("Proceso :"+str(id_proceso)+" rng "+str(rn.integers(low=0,high=10000000000)))
        
        for j in tqdm(range(int(samples/bloques)),desc="Proceso: "+str(id_fichero)):

            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    result=mcsolve(lista_de_hamiltonianos,est,tlist[contador*bloques:(contador+1)*bloques],c_ops=operadores_de_salto,ntraj=1,seeds=rn.integers(low=0,high=10000000000),options={
                            "store_final_state":True,"nsteps":1000000000,"atol":1e-9,"rtol":1e-9,"num_cpus":1,"keep_runs_results":True,"progress_bar":None
                        })

            est=result.final_state[0]
            
            trabajos.append(ex.submit(medidas_bloque2,[i for i in range(contador*bloques,(contador+1)*bloques)],result.states[0],nombres,nombre_fichero_resultados,id_proceso))
            #medidas_bloque2([i for i in range(contador*bloques,(contador+1)*bloques)],result.states[0],resultados,nombres,nombre_fichero_resultados)
            contador+=1

        
        suma=1
        for trab in futures.as_completed(trabajos):
            suma+=trab.result()
        
    except KeyboardInterrupt:
        csv.close()
        csv.unlink()

    
    csv.close()
    csv.unlink()
    return 1


# In[38]:


try:
    ex=futures.ProcessPoolExecutor(max_workers=hilos)
    trabajos=[]
    t1=t.time()
    est=psi0
    contador=0
    for j in range(numero_de_trayectorias):
        trabajos.append(ex.submit(trayectoria,psi0,[H0,[Ht,f],[Ht2,func]],jump,bloques,tlist,rng.randint(0,10000000000),2,nombres,medidas_bloque2,j,rng.randint(0,10000000000)))
        #trabajos.append(ex.submit(mcsolve,[H0,[Ht,f]],est,tlist,c_ops=jump,ntraj=1,options={"store_final_state":True,"nsteps":10000000,"atol":1e-6,"rtol":1e-4,"num_cpus":1,"keep_runs_results":True}))
        #result=mcsolve([H0,[Ht,f]],est,tlist,c_ops=jump,ntraj=1,options={"store_final_state":True,"nsteps":10000000,"atol":1e-6,"rtol":1e-4,"num_cpus":1,"keep_runs_results":True})
        #trayectoria(psi0,[H0,[Ht,f]],jump,bloques,tlist,j,hilos_medida,medidas_bloque2)

    
    progress_bar = tqdm(total=100)
    suma=1
    for trab in futures.as_completed(trabajos):
    
        progress_bar.n=suma*100/float(numero_de_trayectorias)
        progress_bar.refresh()
        suma+=trab.result()
    progress_bar.close()

    
    
    
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


# In[39]:


#result=mcsolve([H0,[Ht,f]],psi0,tlist,c_ops=jump,ntraj=5,options={
#    "store_final_state":True,"nsteps":10000000,"progress_bar":"tqdm","atol":1e-6,"rtol":1e-4,"num_cpus":5,"keep_runs_results":True
#})


# In[40]:


try:
    nombre1="osciladores_disipativos_motecarlo_externo_id_"
    nombre2="_n_osc_"+str(n_oscillators)+"_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+"_J_"+str(J[0,1])+".csv"
    
    
    ls3=[("Prob_conf",i) for i in range(2**n_oscillators)]
    ls4=[("Error",i) for i in range(2**n_oscillators)]
    
    columns=pd.MultiIndex.from_tuples([*ls3,*ls4])
    
    resultados=pd.DataFrame(columns=columns,index=range(samples))
    
    datos=np.zeros((numero_de_trayectorias,samples,2**n_oscillators))
    
    for i in range(numero_de_trayectorias):
        aux=np.genfromtxt(nombre1+str(i)+nombre2,delimiter=",")
        datos[i,:,:]=aux
    
    datos_m=datos.mean(axis=0)
    datos_e=datos.std(axis=0)/np.sqrt(numero_de_trayectorias)
    
    resultados.iloc[:,:2**n_oscillators]=datos_m
    resultados.iloc[:,2**n_oscillators:]=datos_e

    
    resultados.to_csv("osciladores_disipativos_motecarlo_externo_n_osc_"+str(n_oscillators)+"_N_"+str(N)+"_m_"+str(m)+"_delta_"+str(delta[0])+"_gammam_"+str(gammam)+"_gamma_"+str(gamma)+"_eta_"+str(etaf)+"_theta_"+str(theta)+"_n_"+str(n)+"_xi_"+str(xi)+"_tau_"+str(tau)+"_J_"+str(J[0,1])+".csv")
    
    os.system("mkdir "+nombre1+nombre2[:-4])
    
    os.system("mv *_id_*"+nombre2+" "+nombre1+nombre2[:-4])
except:
    pass


# In[ ]:




