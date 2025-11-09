import numpy as np
from numpy.typing import NDArray
from typing import List,Dict
from layer import Layer
# from activation.activation_function import activation_function

class LayerNorm(Layer):
    
    def __init__(self,
                hidden_units:int,
                ):
        self.eps:np.float32 = 1e-5
        self.gamma:NDArray = np.ones(hidden_units)
        self.beta:NDArray = np.zeros(hidden_units)
        
        
        self.cache:Dict = {}
        self.gradients:Dict = {}
        self.hidden_units = hidden_units
    
    def forward_pass(self,x:NDArray)->NDArray:
        x_mean = np.mean(x, axis=-1,keepdims=True)
        x_var = np.var(x, axis=-1, keepdims=True)
        x_std = np.sqrt(x_var + self.eps)
        x_norm = (x - x_mean) / x_std
        
        self.cache['x_norm'] = x_norm
        self.cache['std_dev'] = x_std
        self.cache['x_mean'] = x_mean 
        self.cache['x_var'] = x_var
        return self.gamma * x_norm + self.beta
    
    def backward_pass(self,d_out:NDArray)->NDArray:
        x_norm = self.cache['x_norm']
        
        self.gradients['d_beta'] = np.sum(d_out, axis=0)
        self.gradients['d_gamma'] = np.sum(d_out * x_norm, axis=0)
        
        d_x_norm = d_out * self.gamma 
        d_x_mean = (1 / self.hidden_units) * np.sum(d_x_norm,axis=-1, keepdims=True)
        d_x_std = (1 / self.hidden_units) * np.sum(d_x_norm * x_norm,axis=-1, keepdims=True)
        
        d_x = (1 / self.cache['std_dev']) * (d_x_norm - d_x_mean - x_norm * d_x_std)
        return d_x 
    
    def update(self,lr:np.float32):
        self.gamma -= lr * self.gradients['d_gamma']
        self.beta  -= lr * self.gradients['d_beta']



