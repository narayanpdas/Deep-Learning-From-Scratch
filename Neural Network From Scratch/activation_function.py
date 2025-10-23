import numpy as np
from numpy.typing import NDArray
import math

def sigmoid(x:float)->float:
    return 1 / (1 + math.exp(-x))

def tanh(x:float)->float:
    ex = math.exp(x)
    ex_ = math.exp(-x)
    return ((ex - ex_) / (ex + ex_))

def relu(x: NDArray)->float:
    return np.maximum(0,x)

# Used to Solve Drying Relu problems
def leaky_relu(x: NDArray)->float:
    return np.maximum(0.01*x,x)
# USed for Back-Propagation
def d_leaky_relu(Z: NDArray) -> NDArray:
    """Derivative of Leaky ReLU"""
    dZ = np.ones_like(Z)
    dZ[Z < 0] = 0.01
    return dZ
# Generally use for the last output layer in Multi - Class Clssification Problem
def softmax(x:NDArray)->NDArray:
    _exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return _exps / np.sum(_exps, axis=-1, keepdims=True)









