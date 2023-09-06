
import numpy as np  #importando uma biblioteca

entrada =np.array([1,7,5])
pesos = np.array([0.8, 0.1, 0])

def soma (e,p):
    return e.dot(p) #dot product / produto escalar 
#este 'dot' el faz soma e multiplicação

    
s = soma(entrada, pesos)
print(f'calono das entrada e dos pesso: {s}')

def stepFuction(soma):
    if(soma >1):
        return 1
    return 0

r = stepFuction(s)

print(f'resultado da ativação do neurônio: {r}')
