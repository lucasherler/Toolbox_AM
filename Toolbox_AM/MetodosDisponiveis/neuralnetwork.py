#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:50:30 2022

Adaptado de: https://github.com/jiexunsee/Neural-Network-with-Python/blob/master/3layer.py

@author: bruno
"""

import _util as util
# from numpy import dot
import numpy as np

class NEURALNETWORK(util.Util):
    
    def _init_params(self):  # "Protected" method
        # default values
        super()._init_params()
        self.number_of_hidden_neurons = 100
        self.activation_function = 'sig'
        self.activation_function_output = 'linear'
        self.number_of_epochs = 100
        self.number_of_hidden_layers = 1
        self._accepted_params.extend([
            'number_of_hidden_neurons',
            'number_of_hidden_layers',
            'number_of_epochs',
        ])
        self._accepted_params.remove('use_random_orthogonalization')
        self.input_weight = []
        self._accepted_params_docs['number_of_hidden_neurons'] = 'Número de Neurônios nas Camadas Ocultas da Rede Neural. Padrão: 1000 em cada camada'
        #self._accepted_params_docs['activation_function'] = 'Função de Ativação Escolhida para utilização naS camadas ocultaS. Padrão: Função Sigmoide (sig)'
        #self._accepted_params_docs['activation_function_output'] = 'Função de Ativação Escolhida para utilização naS camadas ocultaS. Padrão: Função Linear'
        self._accepted_params_docs.pop('use_random_orthogonalization') #] = 'Não Utilizado nesse código'
        self._accepted_params_docs['number_of_epochs'] = 'Número de épocas/iterações do treino da rede neural. Padrão: 100'
        self._accepted_params_docs['number_of_hidden_layers'] = 'Número de camadas ocultas da rede neural. Padrão: 1'
        
    def __init__(self, param_dict={}):

        self._init_params()
        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")
        #if self.number_of_input_neurons == []:
        #    raise NameError("Number of input neurons not defined!")
        self.activation_function = self.parse_activation_function(
            self.activation_function)
        self.activation_function_output = self.parse_activation_function(
            self.activation_function_output)
        self.parse_seed(self.seed)  # hw to mk more readable Util.parse_seed?
    
    
#     def __init__(self):
#         random.seed(1)

#         # setting the number of nodes in layer 2 and layer 3
#         # more nodes --> more confidence in predictions (?)
#         l2 = 5
#         l3 = 4

#         # assign random weights to matrices in network
#         # format is (no. of nodes in previous layer) x (no. of nodes in following layer)
#         self.synaptic_weights1 = 2 * random.random((3, l2)) -1
#         self.synaptic_weights2 = 2 * random.random((l2, l3)) -1
#         self.synaptic_weights3 = 2 * random.random((l3, 1)) -1
        
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x*(1-x)

    # train neural network, adusting synaptic weights each time
    def train(self, X, Y):
        
        # %%
        
        
        # # pass training set through our neural network
        # # a2 means the activations fed to second layer
        # a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
        # a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        # output = self.__sigmoid(dot(a3, self.synaptic_weights3))

        # # calculate 'error'
        # del4 = (training_set_outputs - output)*self.__sigmoid_derivative(output)

        # # find 'errors' in each layer
        # del3 = dot(self.synaptic_weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
        # del2 = dot(self.synaptic_weights2, del3)*(self.__sigmoid_derivative(a2).T)

        # # get adjustments (gradients) for each layer
        # adjustment3 = dot(a3.T, del4)
        # adjustment2 = dot(a2.T, del3.T)
        # adjustment1 = dot(training_set_inputs.T, del2.T)
        
        # %%
        self.number_of_input_neurons = X.shape[1]
        self.input_weight = [[]]*(self.number_of_hidden_layers+1) #Apenas alocar a lista com tamanho n+1
        self.input_weight[0] = self.generate_random_weights(
            self.number_of_input_neurons,
            self.number_of_hidden_neurons,
            X)
        for i in range(1,self.number_of_hidden_layers): 
            self.input_weight[i] = self.generate_random_weights(
                self.number_of_hidden_neurons,
                self.number_of_hidden_neurons,
                X)
        self.input_weight[-1] = self.generate_random_weights(
            self.number_of_hidden_neurons,
            Y.shape[1],
            X)
        
        hidden_output = [[]]*(self.number_of_hidden_layers+1)
        errors = [[]]*(self.number_of_hidden_layers+1)
        adjustment = [[]]*(self.number_of_hidden_layers+1)
        # hidden_output[0] = X
        for iteration in range(self.number_of_epochs):
            # %%
            #Forward
            hidden_output[0] = self.activation_function(np.dot(X, self.input_weight[0]))
            for k in range(1,self.number_of_hidden_layers):
                hidden_output[k] = self.activation_function(np.dot(hidden_output[k-1], self.input_weight[k]))
            hidden_output[-1] = self.activation_function_output(np.dot(hidden_output[-2], self.input_weight[-1]))
                
            #Backward
            errors[-1] = (Y - hidden_output[-1])*np.ones(Y.shape)
            errors[-1] = errors[-1].T
            for k in range(self.number_of_hidden_layers-1,-1,-1):
                errors[k] = np.dot(self.input_weight[k+1],errors[k+1])*self.__sigmoid_derivative(hidden_output[k]).T
            
            # get adjustments (gradients) for each layer
            #adjustment[-1] = dot(hidden_output[-1].T,errors[-1])
            for k in range(self.number_of_hidden_layers,0,-1):
                adjustment[k] = np.dot(hidden_output[k-1].T,errors[k].T)
            adjustment[0] = np.dot(X.T,errors[0].T)
            
            for k in range(0,len(self.input_weight)):
                self.input_weight[k] += adjustment[k]


    def predict(self, inputs):
        hidden_output = [[]]*(self.number_of_hidden_layers+1)
        # pass our inputs through our neural network
        hidden_output[0] = self.activation_function(np.dot(X, self.input_weight[0]))
        for k in range(1,self.number_of_hidden_layers):
            hidden_output[k] = self.activation_function(np.dot(hidden_output[k-1], self.input_weight[k]))
        hidden_output[-1] = self.activation_function_output(np.dot(hidden_output[-2], self.input_weight[-1]))
        return hidden_output[-1]

if __name__ == "__main__":
#     # initialise single neuron neural network
    neural_network = NEURALNETWORK()

#     # the training set.
    X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    Y = np.array([[0,1,1,0]]).T

    neural_network.train(X, Y)

    # test with new input
    print("\nConsidering new situation [1,0,0] -> ?")
    print(neural_network.predict(np.array([1,0,0])))
    # Yhat = neural_network.predict(X)
