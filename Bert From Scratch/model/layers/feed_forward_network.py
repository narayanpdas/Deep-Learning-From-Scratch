import numpy as np
from typing import Dict
from numpy.typing import NDArray
from layers.layer import Layer
from layers.dense import Dense
from layers.activation.activation_functions import Linear,Relu





class FFN(Layer):
    def __init__(self,hidden_size:int=768,factor:int=4):
        self.intermediate_size:int=factor * hidden_size
        self.dense1 = Dense(units=self.intermediate_size,
                            input_size=hidden_size,
                            activation_function=Relu())
        self.dense2 = Dense(units=hidden_size, 
                            input_size=self.intermediate_size, 
                            activation_function=Linear())

    def forward_pass(self,x:NDArray):
        out = self.dense1.forward_pass(x=x)
        out = self.dense2.forward_pass(x=out)
        return out
    
    def backward_pass(self,d_out:NDArray):
        d_x = self.dense2.backward_pass(d_out=d_out)
        d_x = self.dense1.backward_pass(d_out=d_x)
        return d_x

    def update(self, lr):
        self.dense1.update(lr=lr)
        self.dense2.update(lr=lr)
        

