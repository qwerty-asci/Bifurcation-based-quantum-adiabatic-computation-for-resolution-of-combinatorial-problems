import numpy as np

rng=np.random

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
eigen=np.array([rng.uniform(3,7),rng.uniform(0,3)])


# print("eigen: ",eigen)

norm1=v1[0]**2+v1[1]**2
norm2=v2[0]**2+v2[1]**2


v1=v1[np.newaxis]
v2=v2[np.newaxis]
M=eigen[0]*v1.T@v1/norm1+eigen[1]*v2.T@v2/norm2


print("M: \n",M)

autovalor=np.linalg.eigvals(M).max()
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

# print(indices_de_vectores)

