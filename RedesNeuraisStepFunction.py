
entrada = [-1,7,5]
pesos = [0.8, 0.1, 0]

def soma (e,p):
    s =0
    for i  in range(3):
       # print(f'entarda: {entrada[i]}')
      #print(f'peso : {pesos[i]}')

        s += e[i] * p[i]#Calculo neural
    return s

    
s = soma(entrada, pesos)
print(f'calono das entrada e dos pesso: {s}')

def stepFuction(soma):
    if(soma >1):
        return 1
    return 0

r = stepFuction(s)

print(f'resultado da ativação do neurônio: {r}')
