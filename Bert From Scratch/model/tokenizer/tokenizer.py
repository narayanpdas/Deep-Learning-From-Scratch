import numpy as np
from numpy.typing import NDArray
from typing import List,Dict

path = "vocab.txt" # For Testing Within the Module


class Tokenizer():
    def __init__(self,path:str,max_len:int=64):
        self.vocab,self.inv_vocab = self._read_vocab(path)
        self.max_len = max_len
        self.input_ids = []
        self.input_ids_nd = np.full(max_len,fill_value=0,dtype=np.int32)
        self.attention_mask_nd = np.full(max_len,fill_value=0,dtype=np.float32)
        self.token_type_ids_nd = np.full(max_len,fill_value=0,dtype=np.int32)
    def _read_vocab(self,path:str):
        with open(path,mode="r",encoding="UTF-8") as f:
            vocab_raw = f.read()
            vocab_dict = {val:idx for idx,val in enumerate(vocab_raw.split())}
            vocab_dict_inv = {idx:val for idx,val in enumerate(vocab_raw.split())}
            f.close() 
        return vocab_dict,vocab_dict_inv
    def _split(self,call:List[str]):
        call[0] = "".join(['[CLS] ' , call[0]])
        if len(call)>1:
            p = [("".join([cl,' [SEP]'])).split(' ') for cl in call]
            final_tokens = [item for tokens in p for item in tokens]
        else:
            p = (call[0].split(' '))
            final_tokens = p
        return final_tokens
    def _recurse(self,token:str):
        key = self.vocab.get(token,None)
        if key:
            self.input_ids.append(key)
            return key
        else:
            for i in reversed(range(len(token))):
                end = token[i:]
                start = token[0:i]
                token_val = self.vocab.get(start,None)
                if token_val:
                    self.input_ids.append(token_val) 
                    return self._recurse(token=f'##{end}')            
    def _break_token(self,tokens:List[str]):
        for token in tokens:
            token_val = self.vocab.get(token,None)
            if token_val:
                self.input_ids.append(token_val)
            else:
                # print(f'Recurese Token: {token}')
                self._recurse(token)
    
    def tokenize(self,sent:List[str]):
        tokens = self._split(call = sent)
        self._break_token(tokens=tokens) 
        padding = min(len(self.input_ids),self.max_len)
        self.input_ids_nd[:padding] = self.input_ids[:padding]
        self.attention_mask_nd[:padding] = 1 if self.input_ids[:padding] else 0
        flag = 0
        for idx,x in enumerate(self.input_ids):
            if idx >= self.max_len: 
                break
            self.token_type_ids_nd[idx] = flag
            if self.inv_vocab.get(x) == '[SEP]':
                flag = abs(flag - 1)
