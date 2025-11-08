import numpy as np
from numpy.typing import NDArray
from typing import List,Dict
from tokenizer.tokenizer import Tokenizer

HIDDEN_SIZE = 768
VOCAB_SIZE = 30000
max_token_len = 32
class EmbeddingLayer():
    def __init__(self):
        self.token_embedding_layer = np.random.normal(loc=0.0,
                                                    scale=0.02,
                                                    size=(VOCAB_SIZE,HIDDEN_SIZE))
        self.segment_embedding_layer = np.random.normal(loc=0.0,
                                                        scale=0.02,
                                                        size=(2,HIDDEN_SIZE)) # Because we are making BERT which Primarily compares 2 Sentences hence  | size (2,768)
        self.postional_embedding_layer = np.random.normal(loc=0.0,
                                                        scale=0.02,
                                                        size=(max_token_len,HIDDEN_SIZE))
        
    def forward_pass(self,input_ids:NDArray,token_type_ids:NDArray):
        inputid_ws = self.token_embedding_layer[input_ids]
        tokentypeids_ws = self.segment_embedding_layer[token_type_ids]
        positinal_ws = self.postional_embedding_layer[np.arange(max_token_len)]
        return inputid_ws + tokentypeids_ws + positinal_ws
    
    def layer_norm(self):
        pass
        
inputt = ['hellow world','ieu wowqden']
tokenizer = Tokenizer("Bert From Scratch/model/tokenizer/vocab.txt",max_len=32)
tokenizer.tokenize(sent=inputt)
layer1 = EmbeddingLayer()
val = layer1.forward_pass(tokenizer.input_ids_nd,
                        tokenizer.token_type_ids_nd,
                        )
print(val.shape)






