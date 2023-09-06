import numpy as np
def sigmoid(soma):
    return 1/(1 + np.exp(-soma)) #metodos exp para Para poder fazer o exponencial

a = sigmoid(1.5)

print(f'Respota: {a}')

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])

pesos1 = np.array([[-0.017],[-0.893], [0.148]])

epocas = 100 #para  o ajute de  pesoa quantas vese  rodar para achar  o ajute de  peso

for j in range(epocas):
  camadaEntrada = entradas
  somaSinapse0 = np.dot(camadaEntrada, pesos0) #mehando os valores  entrada Multiplicando pelo peso
  camadaOculta = sigmoid(somaSinapse0)

print(somaSinapse0)
print(f'Camada oculta: {camadaOculta}')