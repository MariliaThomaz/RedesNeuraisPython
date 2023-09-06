import numpy as np
def sigmoid(soma):
    return 1/(1 + np.exp(-soma)) #metodos exp para Para poder fazer o exponencial


def sigmoidDerivada(sig):
   return sig * (1-sig)

#a = sigmoid(0.5)
#b = sigmoidDerivada(a)

#print(f'Respota: {a}')

#print(f'Sigmoid Derivada: {b}')

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
saidas = np.array([[0],[1],[1],[0]])

# pesos0 = np.array([[-0.424, -0.740, -0.961],
#                    [0.358, -0.577, -0.469]])

#pesos1 = np.array([[-0.017],[-0.893], [0.148]])
pesos1 = 2*np.random.random((3,1)) -1

#crinado pessoa altomaticos
pesos0 = 2*np.random.random((2,3)) -1

epocas = 100 #para  o ajute de  pesoa quantas vese  rodar para achar  o ajute de  peso
taxaApredizagem = 0.6
momento =1

for j in range(epocas):
  camadaEntrada = entradas
  somaSinapse0 = np.dot(camadaEntrada, pesos0) #mehando os valores  entrada Multiplicando pelo peso
  camadaOculta = sigmoid(somaSinapse0)

  somaSinapse1 = np.dot(camadaOculta, pesos1)
  canadaSaida = sigmoid(somaSinapse1)

  #calulando para ver erro
  erroCamdaSaida = saidas - canadaSaida

  #media dos  erros
  mediaAbsoluta = np.mean(np.abs(erroCamdaSaida))
  print(f'Erro: {mediaAbsoluta}')

  #calculando  a Derivada
  derivadaSaida = sigmoidDerivada(canadaSaida)

  deltaSaida = erroCamdaSaida * derivadaSaida
#paara faz  isto matriz transposta
  pesos1Transpota = pesos1.T
  deltaSaidaXpeso= deltaSaida.dot(pesos1Transpota)

  deltaCamadaOculta =  deltaSaidaXpeso * sigmoidDerivada(camadaOculta)

  camadaOcultaTranspota = camadaOculta.T
  pessoaNovo1 = camadaOcultaTranspota.dot(deltaSaida)
  pesos1 = (pesos1 * momento) + (pessoaNovo1 * taxaApredizagem)

  camadaEntradaTranspota = camadaEntrada.T
  pessoaNovo0 = camadaEntradaTranspota.dot(deltaCamadaOculta)
  pesos0 = (pesos0  * momento) + (pessoaNovo0 * taxaApredizagem)
  

print(f'Soma Sinapese o:\n{somaSinapse0}')
print(f'Camada oculta:\n {camadaOculta}')
print(f'Soma Sinapse 1:\n {somaSinapse1}')
print(f'Camada Saida:\n {canadaSaida}')
print(f'Erro Camada Saida:\n {erroCamdaSaida}')
print(f'Media abisoluta: {mediaAbsoluta}')
print(f'Derivada de Saida:\n {derivadaSaida}')
print(f'Delta Saida:\n {deltaSaida}')
print(f'Delta Saida veses Peso:\n {deltaSaidaXpeso}')
print(f'Delta camada Oculta: \n {deltaCamadaOculta}')
print(f'Pesoas 1:\n {pessoaNovo1}')
print(f'pesos 1\n {pesos1}')
print(f'Pesoas 0:\n {pesos0}')
