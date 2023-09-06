import numpy as np

entrada = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,0,0,1])
pesos = np.array([0.0,0.0])

taxaApredizagem = 0.1

def stepFuction(soma):
    if(soma >1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFuction(s)

def treinar():
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal =0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entrada[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos [j] + (taxaApredizagem * entrada[i][j] * erro)
                print(f'Peso atualizado: {pesos[j]}')
        print(f'Toal de erros: {erroTotal}')

treinar()