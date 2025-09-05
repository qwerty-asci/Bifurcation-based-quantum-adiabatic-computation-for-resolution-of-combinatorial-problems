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

q=2
tau=200


M=np.zeros((2,2))

x1=0
x2=0

while(x1==0 and x2==0):
    num=rng.randint(0,2**4)
    autovector_propio_solucion=num

    cadena=str(bin(num))[2:]


    cadena=(4-len(cadena))*"0"+cadena

    # print(cadena)

    x1=0
    x2=0

    if(cadena[0]=="1"):
        x1+=1
        x1<<=1
    if(cadena[1]=="1"):
        x1+=1

    if(cadena[2]=="1"):
        x2+=1
        x2<<=1
    if(cadena[3]=="1"):
        x2+=1
    #
    # print("num: ",num," bin: ",bin(num))
    # print("x1: ",x1," bin: ",bin(x1))
    # print("x2: ",x2," bin: ",bin(x2))

v1=np.array([x1,x2])

y1=rng.randint(1,4)
try:
    y2=-x1*y1/x2
except:
    y2=rng.randint(1,4)
    y2=-x2*y2/x1

if(x1==0 or x2==0):
    y1=0
    y2=0
    while(y1==0 and y2==0):
        if(x2==0):
            y1=0
            y2=rng.randint(1,4)
            continue
        if(x1==0):
            y2=0
            y1=rng.randint(1,4)
            continue

v2=np.array([y1,y2])
#
# print("v1: ",v1)
# print("v2: ",v2)

aleatorio=rng.uniform(0.01,17)
eigen=np.array([aleatorio,rng.uniform(0,aleatorio)])


# print("eigen: ",eigen)

norm1=v1[0]**2+v1[1]**2
norm2=v2[0]**2+v2[1]**2


v1=v1[np.newaxis]
v2=v2[np.newaxis]
M=eigen[1]*v1.T@v1/norm1+eigen[0]*v2.T@v2/norm2


print("M: \n",M)

autovalor=eigen[1]
print("Autovalores: ",eigen)
print("Autovectores-1: ",v1)
print("Autovectores-2: ",v2)



indices_de_vectores=[2**4-1-autovector_propio_solucion]

#Compruebo cuales son los vectores proporcionales

# cadena="0100"

#Incremento
#por 2
test1=2*(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))
test2=2*(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))

# print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
# print("tests: ",test1," ",test2)
if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
    cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
    indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
    # print("indice_2:",2**4-1-indice_2)

    # print("Cadena_2: ",cadena_por_2)

    indices_de_vectores.append(2**4-1-indice_2)

#por 3
test1=3*(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))
test2=3*(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))

# print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
# print("tests: ",test1," ",test2)
if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
    cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
    indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
    # print("indice_2:",2**4-1-indice_2)

    # print("Cadena_2: ",cadena_por_2)

    indices_de_vectores.append(2**4-1-indice_2)






#Decremento
#entre 2
test1=(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))/2
test2=(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))/2
if((test1-int(test1))==0 and (test2-int(test2))==0):
    test1=int(test1)
    test2=int(test2)
    # print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
    # print("tests: ",test1," ",test2)
    if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
        cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
        indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
        # print("indice_2:",2**4-1-indice_2)

        # print("Cadena_2: ",cadena_por_2)

        indices_de_vectores.append(2**4-1-indice_2)


#entre 3
test1=(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))/3
test2=(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))/3
if((test1-int(test1))==0 and (test2-int(test2))==0):
    test1=int(test1)
    test2=int(test2)
    # print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
    # print("tests: ",test1," ",test2)
    if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
        cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
        indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
        # print("indice_2:",2**4-1-indice_2)

        # print("Cadena_2: ",cadena_por_2)

        indices_de_vectores.append(2**4-1-indice_2)


test1=(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))/1.5
test2=(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))/1.5
if((test1-int(test1))==0 and (test2-int(test2))==0):
    test1=int(test1)
    test2=int(test2)
    # print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
    # print("tests: ",test1," ",test2)
    if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
        cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
        indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
        # print("indice_2:",2**4-1-indice_2)

        # print("Cadena_2: ",cadena_por_2)

        indices_de_vectores.append(2**4-1-indice_2)


test1=(2*(1 if cadena[0]=="1" else 0)+(1 if cadena[1]=="1" else 0))*1.5
test2=(2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))*1.5
if((test1-int(test1))==0 and (test2-int(test2))==0):
    test1=int(test1)
    test2=int(test2)
    # print("indice_1:",2**4-1-((8*(1 if cadena[0]=="1" else 0)+4*(1 if cadena[1]=="1" else 0)+2*(1 if cadena[2]=="1" else 0)+(1 if cadena[3]=="1" else 0))))
    # print("tests: ",test1," ",test2)
    if(len(str(bin(test1))[2:])<=2 and len(str(bin(test2))[2:])<=2):
        cadena_por_2=(2-len(str(bin(test1))[2:]))*"0"+str(bin(test1))[2:]+(2-len(str(bin(test2))[2:]))*"0"+str(bin(test2))[2:]
        indice_2=(8*(1 if cadena_por_2[0]=="1" else 0)+4*(1 if cadena_por_2[1]=="1" else 0)+2*(1 if cadena_por_2[2]=="1" else 0)+(1 if cadena_por_2[3]=="1" else 0))
        # print("indice_2:",2**4-1-indice_2)

        # print("Cadena_2: ",cadena_por_2)

        indices_de_vectores.append(2**4-1-indice_2)


Md=np.conj(M.T)

M2=Md@M



a=[]
ad=[]

for i in range(n_oscillators):
    delta[i]=2*K

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


av=[]
aux=[]
avd=[]
aux2=[]
cont=0
for i in range(len(a)):

    aux.append(a[i])
    aux2.append(ad[i])
    cont+=1
    if(cont%q==0 and cont!=0):
        av.append(aux)
        avd.append(aux2)
        aux2=[]
        aux=[]


cont=0
psi0=basis(N**n_oscillators,0)

tlist=np.linspace(0.0,tmax/K,samples)

H0=qzero(N**n_oscillators)
ising=qzero(N**n_oscillators)
Ht=qzero(N**n_oscillators)
H1=qzero(N**n_oscillators)
H2=qzero(N**n_oscillators)
H3=qzero(N**n_oscillators)

H4=qzero(N**n_oscillators)
H5=qzero(N**n_oscillators)
H6=qzero(N**n_oscillators)

#f=lambda t:-t*K*K*pf
f=lambda t: -pf*K*K*np.tanh(t/tau)
def g(t):

    if(t<=1000):
        return 0.0
    else:
        return 0.05*(1-np.tanh((t-1000)/200))

alpha=lambda t:(-f(t)-delta[0]*np.tanh(-f(t)/(delta[0])))*xi/K
beta=lambda t:np.sqrt((-f(t)-delta[0]*np.tanh(-f(t)/(delta[0])))/K)*xi/2.0
gamma=lambda t:xi/4.0

alpha2=lambda t:(-f(t)-delta[0]*np.tanh(-f(t)/(delta[0])))*xi*g(t)/K
beta2=lambda t:np.sqrt((-f(t)-delta[0]*np.tanh(-f(t)/(delta[0])))/K)*xi*g(t)/2.0
gamma2=lambda t:-xi*g(t)/4.0


for i in range(n_oscillators):
    H0+=delta[i]*ad[i]*a[i]+K*ad[i]*ad[i]*a[i]*a[i]/2.0
    Ht+=(ad[i]*ad[i]+a[i]*a[i])/2.0


#Termino independiente
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H1+=2**(2*q-k-l-4)*qeye(N**n_oscillators)*autovalor**2

            for j in range(M.shape[0]):
                H1+=2**(2*q-k-l-4)*qeye(N**n_oscillators)*(M2[i,j]-autovalor*M[i,j]-autovalor*Md[i,j])

#Termino lineal
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H2+=2*2**(2*q-k-l-4)*(av[i][k]+avd[i][k])*autovalor**2

            for j in range(M.shape[0]):
                H2+=2**(2*q-k-l-4)*(av[i][k]+avd[i][k]+av[j][l]+avd[j][l])*(M2[i,j]-autovalor*M[i,j]-autovalor*Md[i,j])

#Termino Ising
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H3+=2**(2*q-k-l-4)*(av[i][k]+avd[i][k])*(av[i][l]+avd[i][l])*autovalor**2
            for j in range(M.shape[0]):
                H3+=2**(2*q-k-l-4)*(av[i][k]+avd[i][k])*(av[j][l]+avd[j][l])*(M2[i,j]-autovalor*M[i,j]-autovalor*Md[i,j])



# #TERMINO DE CORRECCION

#Termino independiente
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H4-=2**(2*q-k-l-4)*qeye(N**n_oscillators)

#Termino lineal
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H5-=2*2**(2*q-k-l-4)*(av[i][k]+avd[i][k])

#Termino Ising
for k in range(q):
    for l in range(q):
        for i in range(M.shape[0]):
            H6-=2**(2*q-k-l-4)*(av[i][k]+avd[i][k])*(av[i][l]+avd[i][l])




ls3=[("Prob_conf",i) for i in range(2**n_oscillators)]

columns=pd.MultiIndex.from_tuples(ls3)

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

def medidas_bloque2(fila,estado,archivo,nombres,A):



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
            archivo.to_csv("eigen_vectors2_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(A[0,0])+"_M21_"+str(A[1,0])+"_M12_"+str(A[0,1])+"_M22_"+str(A[1,1])+"_tau_"+str(tau)+".csv")
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
            archivo.to_csv("eigen_vectors2_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(A[0,0])+"_M21_"+str(A[1,0])+"_M12_"+str(A[0,1])+"_M22_"+str(A[1,1])+"_tau_"+str(tau)+".csv")
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

        result=sesolve([H0,[Ht,f],[H1,alpha],[H2,beta],[H3,gamma],[H4,alpha2],[H5,beta2],[H6,gamma2]],est,tlist[contador*bloques:(contador+1)*bloques],options={
            "store_final_state":True,"store_states":True,"nsteps":10000000,"normalize_output":False,"atol":1e-8,"rtol":1e-8
        })


        est=result.final_state
        trabajos.append(ex.submit(medidas_bloque2,[i for i in range(contador*bloques,(contador+1)*bloques)],result.states,resultados,nombres,M))
        contador+=1

    progress_bar = tqdm(total=100)
    suma=1
    for trab in futures.as_completed(trabajos):

        progress_bar.n=suma*100/float(samples/bloques)
        progress_bar.refresh()
        suma+=trab.result()
    progress_bar.close()
    resultados.to_csv("eigen_vectors2_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(M[0,0])+"_M21_"+str(M[1,0])+"_M12_"+str(M[0,1])+"_M22_"+str(M[1,1])+"_tau_"+str(tau)+".csv")

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

resultados.to_csv("eigen_vectors2_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(M[0,0])+"_M21_"+str(M[1,0])+"_M12_"+str(M[0,1])+"_M22_"+str(M[1,1])+"_tau_"+str(tau)+".csv")


resultado_archivo=open("eigen_vectors2_K_"+str(K)+"_pf_"+str(pf)+"_xi_"+str(xi)+"_tmax_"+str(tmax)+"_n_osciladores_"+str(n_oscillators)+"_N_"+str(N)+"_g_"+str(g)+"_M11_"+str(M[0,0])+"_M21_"+str(M[1,0])+"_M12_"+str(M[0,1])+"_M22_"+str(M[1,1])+"_tau_"+str(tau)+".dat","w")
resultado_archivo.write(str(resultados["Prob_conf"].to_numpy()[-1,np.array(indices_de_vectores)].sum())+" "+str(np.abs(eigen[0]-eigen[1])))
resultado_archivo.close()


