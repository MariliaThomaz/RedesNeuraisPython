from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris  = datasets.load_iris()
entradas = iris.data
saidas = iris.target

# Criado a rede neural
redeNeural = MLPClassifier(verbose=True,
                            max_iter=1000,
                            tol=0.00001,
                            learning_rate=0.3) 

#Usando o metro fit para encaixar
redeNeural.fit(entradas, saidas)
redeNeural.predict([[5,7.2, 5.1, 2.2]])
print(f'Método fit: \n {redeNeural}')
