import numpy as np
from typing import List, Dict,Tuple
from numpy.typing import NDArray
from layers.activation.activation_functions import activation_function
from layers.layer import Layer

class Dense(Layer):
    def __init__(self,
                units:int,
                input_size:int,
                activation_function:activation_function,
                
        ):
        self.units = units
        self.input = input_size
        self.weights = np.random.randn(input_size,units) * np.sqrt(2.0 / input_size) # Scalling for better init
        self.biases = np.zeros(units)
        self.activation_function = activation_function

        self.cache:Dict = {}
        self.gradients:Dict = {}
        
    
    def forward_pass(self,x:NDArray)->NDArray:
        self.cache['A_prev'] = x
        z = np.dot(x,self.weights) + self.biases
        self.cache['z'] = z
        a = self.activation_function.activate(z)
        return a
    
    def backward_pass(self,d_out):
        a_prev = self.cache['A_prev']
        z = self.cache['z']
        m = a_prev.shape[0]
        dz = d_out * self.activation_function.d_activate(z)
        
        dw = (1/m) * np.dot(a_prev.T, dz)
        db = (1/m) * np.sum(dz,axis=0,keepdims=True)
        self.gradients['dw'] = dw
        self.gradients['db'] = db
        
        da = np.dot(dz, self.weights.T)
        
        return da
    
    def update(self, lr:np.float32):
        self.weights -= lr * self.gradients['dw']
        self.biases  -= lr * self.gradients['db']
        
    def _get_layer(self):
        return self.weights,self.biases
    
    def __repr__(self):
        return f"Layer(unit:{self.units} input_size = {self.input})"
    