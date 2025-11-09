import numpy as np
from numpy.typing import NDArray
from typing import List,Dict
from tokenizer.tokenizer import Tokenizer
from layer import Layer

HIDDEN_SIZE = 768
VOCAB_SIZE = 30000
max_token_len = 32
class EmbeddingLayer(Layer):
    def __init__(self,max_token_len:int=max_token_len,
                vocab_size:int=VOCAB_SIZE,
                hidden_units:int=HIDDEN_SIZE):
        self.token_embedding_layer = np.random.normal(loc=0.0,
                                                    scale=0.02,
                                                    size=(vocab_size,hidden_units))
        self.segment_embedding_layer = np.random.normal(loc=0.0,
                                                        scale=0.02,
                                                        size=(2,hidden_units)) # Because we are making BERT which Primarily compares 2 Sentences hence  | size (2,768)
        self.postional_embedding_layer = np.random.normal(loc=0.0,
                                                        scale=0.02,
                                                        size=(max_token_len,hidden_units))
        self.hidden_units = hidden_units
        self.max_token_len = max_token_len
        self.vocab_size = vocab_size
        self.cache:Dict = {}
        self.gradients:Dict = {}
    def forward_pass(self,input_ids:NDArray,token_type_ids:NDArray):
        self.cache['input_ids'] = input_ids
        self.cache['token_type_ids'] = token_type_ids
        
        inputid_ws = self.token_embedding_layer[input_ids]
        tokentypeids_ws = self.segment_embedding_layer[token_type_ids]
        positinal_ws = self.postional_embedding_layer[np.arange(max_token_len)]
        return inputid_ws + tokentypeids_ws + positinal_ws
    
    def backward_pass(self,d_out:NDArray):
        dw_inputids = np.zeros(shape=(self.vocab_size,self.hidden_units))
        dw_typeids = np.zeros(shape=(2,self.hidden_units))
        dw_posids = np.zeros(shape=(self.max_token_len,self.hidden_units))
        
        np.add.at(dw_inputids,self.cache['input_ids'],d_out)
        np.add.at(dw_typeids,self.cache['token_type_ids'],d_out)
        np.add.at(dw_posids,np.arange(self.max_token_len),d_out)
        
        self.gradients['dw_inputids'] = dw_inputids
        self.gradients['dw_typeids'] = dw_typeids
        self.gradients['dw_posids'] = dw_posids
        
        return None
    
    def update(self, lr:np.float32):
        self.token_embedding_layer -= lr * self.gradients['dw_inputids']
        self.segment_embedding_layer -= lr * self.gradients['dw_typeids']
        self.postional_embedding_layer -= lr * self.gradients['dw_posids']
        
        
inputt = ['hellow world','ieu wowqden']
tokenizer = Tokenizer("Bert From Scratch/model/tokenizer/vocab.txt",max_len=32)
tokenizer.tokenize(sent=inputt)
layer1 = EmbeddingLayer()
val = layer1.forward_pass(tokenizer.input_ids_nd,
                        tokenizer.token_type_ids_nd,
                        )
print(val.shape,val)






