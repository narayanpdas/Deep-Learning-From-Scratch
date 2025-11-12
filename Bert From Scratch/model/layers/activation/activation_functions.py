import numpy as np
from numpy.typing import NDArray
from abc import ABC , abstractmethod
from typing import Dict



class activation_function(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def activate(self,x: NDArray):
        pass
    @abstractmethod
    def d_activate(self,):
        pass

class Relu(activation_function):
    def __init__(self):
        pass
    def activate(self,x: NDArray):
        return np.maximum(0,x)
    def d_activate(self, z:NDArray):
        dz = np.ones_like(z)
        dz[z<=0] = 0
        return dz
    
class Leaky_Relu(activation_function):
    def __init__(self,b:np.float32=0.01):
        self.b = b
    def activate(self,x: NDArray):
        return np.maximum(0,x)
    def d_activate(self, z:NDArray):
        dz = np.ones_like(z)
        dz[z<=0] = self.b
        return dz

class Softmax_WD(activation_function):
    # Made keeping in Mind for the use case in attention only models.
    def __init__(self):
        self.cache:Dict= {}
        pass
    def activate(self,x: NDArray):
        _exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.cache['A'] = _exps / np.sum(_exps, axis=-1, keepdims=True)
        return self.cache['A'] 
    def d_activate(self,d_A:NDArray):
        A = self.cache['A'] 
        sum_part = np.sum(d_A * A, axis=-1, keepdims=True)
        d_Z = A * (d_A - sum_part)
        return d_Z

class Linear(activation_function):
    def __init__(self):
        pass
    def activate(self, x:NDArray):
        return x
    def d_activate(self,d_out:NDArray):
        return np.ones_like(d_out)
        
# Generally use for the last output layer in Multi - Class Clssification Problem
def softmax(x:NDArray)->NDArray:
    _exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return _exps / np.sum(_exps, axis=-1, keepdims=True)

def sigmoid(x:float)->float:
    return 1 / (1 + np.exp(-x))







