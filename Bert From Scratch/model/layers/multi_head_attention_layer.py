import numpy as np
from typing import Dict
from layers.layer import Layer
from numpy.typing import NDArray
from layers.dense import Dense
from layers.single_head_attentionlayer import Attention_layer
from layers.activation.activation_functions import Linear


class Multi_Head_Attention_Layer(Layer):
    def __init__(self,head_units:int=12,head_dim:int=64):
        
        self.head_units = head_units
        self.heads = [Attention_layer(head_dim=head_dim,hidden_size=head_units*head_dim) for _ in range(head_units)]
        self.W_o = Dense(units = head_units * head_dim,
                        input_size = head_units * head_dim,
                        activation_function=Linear())
        self.cache:Dict = {}
        
    def forward_pass(self,x:NDArray,attention_mask:NDArray):
        head_outputs = []
        self.cache['x'] = x
        for head in self.heads:
            output = head.forward_pass(x=x,attention_mask=attention_mask)
            head_outputs.append(output)
        output = np.concatenate(head_outputs,axis=-1) # Stiching the head Outputs
        return self.W_o.forward_pass(output)
    
    def backward_pass(self,d_out):
        d_concatenated = self.W_o.backward_pass(d_out)
        d_head_outputs = np.array_split(d_concatenated, self.head_units , axis=-1)
        total_d_x = np.zeros_like(self.cache['x'])
        for i in range(self.head_units):
            head = self.heads[i]
            d_head_output = d_head_outputs[i]
            d_x_head = head.backward_pass(d_head_output)
            total_d_x += d_x_head
        return total_d_x
    
    def update(self, lr):
        self.W_o.update(lr)
        for head in self.heads:
            head.update(lr=lr)