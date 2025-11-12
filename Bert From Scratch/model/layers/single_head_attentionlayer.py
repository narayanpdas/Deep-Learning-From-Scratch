import numpy as np
from typing import List, Dict
from numpy.typing import NDArray
from layers.layer import Layer
from layers.dense import Dense
from layers.activation.activation_functions import Linear,Softmax_WD




class Attention_layer(Layer):
    def __init__(self,head_dim:np.int32=64,hidden_size:np.int32=768):
        self.head_dim = head_dim
        self.W_q = Dense(units=head_dim,input_size=hidden_size,activation_function=Linear())
        self.W_k = Dense(units=head_dim,input_size=hidden_size,activation_function=Linear())
        self.W_v = Dense(units=head_dim,input_size=hidden_size,activation_function=Linear())
        self.softmax = Softmax_WD() # works as a normal activation but mentioned as a layer here
        
        self.cache:Dict = {}
        self.gradients:Dict = {}
    def forward_pass(self,x:NDArray,attention_mask:NDArray):
        Q = self.W_q.forward_pass(x)
        K = self.W_k.forward_pass(x)
        V = self.W_v.forward_pass(x)
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        
        score = np.dot(Q,K.T)
        scaled_scores = score / np.sqrt(self.head_dim)
        
        attention_mask_scaled = attention_mask.reshape(1,-1)
        attention_mask_scaled = np.where(attention_mask_scaled == 0,-1e9,0)
        
        masked_scores = scaled_scores + attention_mask_scaled
        weights = self.softmax.activate(masked_scores)
        self.cache['weights'] = weights
        
        output = np.dot(weights, V)
        
        return output
    
    def backward_pass(self,d_out:NDArray):

        Q = self.cache['Q'] 
        K = self.cache['K'] 
        V = self.cache['V']
        weights:NDArray = self.cache['weights']
        
        d_weights = np.dot(d_out, V.T)
        d_masked_scores = self.softmax.d_activate(d_weights)
        d_scaled_scores = d_masked_scores
        d_scores:NDArray = d_scaled_scores / np.sqrt(self.head_dim)
        
        d_v = np.dot(weights.T, d_out)
        d_Q = np.dot(d_scores.T,K)
        d_K = np.dot(d_scores.T,Q)
                
        d_x_q = self.W_q.backward_pass(d_out=d_Q)
        d_x_k = self.W_k.backward_pass(d_out=d_K)
        d_x_v = self.W_v.backward_pass(d_out=d_v)
        d_x = d_x_q + d_x_k + d_x_v
        
        return d_x
    
    def update(self, lr):
        self.W_q.update(lr=lr)
        self.W_k.update(lr=lr)
        self.W_v.update(lr=lr)
        