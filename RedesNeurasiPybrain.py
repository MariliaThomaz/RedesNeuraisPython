from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

 

rede = FeedForwardNetwork()

 

camadaEntrada = LinearLayer(4)  # Alterado para 4 em vez de 2, para corresponder à entrada da rede
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)

 

bias1 = BiasUnit()
bias2 = BiasUnit()

 

# Adicione objetos à rede neural
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

 

# Ligações entre as camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

 

# Efetivamente a rede neural será construída
rede.sortModules()

 

print(f'Deu certo {rede}')