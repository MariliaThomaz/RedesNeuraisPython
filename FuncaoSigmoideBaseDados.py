import numpy as np
from sklearn import datasets
def sigmoid(soma):
    return 1/(1 + np.exp(-soma)) #metodos exp para Para poder fazer o exponencial


def sigmoidDerivada(sig):
   return sig * (1-sig)

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569, 1], dtype=int)

for i in range(569):
   saidas[i] = valoresSaida[i]


pesos0 = 2*np.random.random((30,3)) -1
pesos1 = 2*np.random.random((3,1)) -1

epocas = 10 #para  o ajute de  pesoa quantas vese  rodar para achar  o ajute de  peso
taxaApredizagem = 0.3
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
  
'''
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
'''