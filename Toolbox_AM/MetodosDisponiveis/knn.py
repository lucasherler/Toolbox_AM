 
#import time
import numpy as np
import _util as util


class KNN(util.Util):
    def _init_params(self):
        super()._init_params()
        self.X = []
        # self.Y = []
        self.K = 1
        self.numOuts = -1
        #self.regularization_parameter = 1000
        #self.alpha = 0
        self._accepted_params.extend(['K'])
        self._accepted_params_docs['K'] = 'Número de vizinhos mais próximos a serem considerados. Padrão: 1'


    def __init__(self, param_dict={}):
        self._init_params()
        self.isRegressionMethod = False
        
    def train(self,X,Y):
        if len(self.X) != 0:
            self.X = np.vstack(self.X,X)
            # self.Y = np.vstack(self.Y,Y)
            self.Yargmax = np.vstack(self.Yargmax, np.argmax(Y,axis=1))
        else:
            self.X = X
            # self.Y = Y
            #Converte de one-hot armazenado pra nº da coluna
            self.Yargmax = np.argmax(Y,axis=1)
            self.numOuts = Y.shape[1]
        pass
    
    def predict(self,X):
        
        #Cria a variável de saída
        out = np.zeros((len(X),self.numOuts))
        
        for i,p in enumerate(X):
            #p/ cada amostra de teste, calcula a distancia para todas as 
            #amostras armazenadas
            ed = np.sqrt(np.sum((p-self.X)**2,axis=1))
            #encontra as k amostras mais próximas
            idx = np.argpartition(ed, self.K) 
            #conta as classes que mais apareceram nos índices das amostras
            #mais próximas (e os valores das classes)
            vals,counts = np.unique(self.Yargmax[idx[:self.K]], return_counts=True)
            #pega a classe que mais apareceu
            idx2 = counts.argmax()
            #marca 1 na variável "out", na coluna correspondente a classe que 
            #mais apareceu nos cálculos
            out[i,vals[idx2]] = 1
            
        return out
        
            
        
            
if __name__ == "__main__":
    knn = KNN()
    # X = np.asarray([[1,2],[3,4],[-10,-10]])
    # Y = np.asarray([[0,1,0],[1,0,0],[0,0,1]])
    # X2 = np.asarray([[1,1],[4,4]])
    # knn.train(X,Y);
    # abc = knn.predict(X2)
    a = np.random.randint(0,2,(150,1))

    aa = np.zeros((150,2))

    for i in range(0,150):
        aa[i,a[i]] = 1

    X = np.random.random((150,4))
    Y = aa
    knn.train(X,Y);
    abc = knn.predict(X)
