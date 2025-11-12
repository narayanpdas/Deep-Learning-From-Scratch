from numpy.typing import NDArray
import numpy as np
import pandas as pd
import ast
class DataLoader():
    def __init__(self,path:str,vocab:dict={},vocab_size:int=30000):
        self.df = pd.read_parquet(path)
        self.raw_text = self.df['raw_text']
        self.input_ids = self.df['input_ids'].to_numpy()
        self.token_type_ids = self.df['token_type_ids'].to_numpy()
        self.attention_mask = self.df['attention_mask'].to_numpy()
        self.vocab = vocab
        self.vocab_size = vocab_size
        pass
    def __len__(self)->int:
        return len(self.input_ids)
    def load_x(self,idx:int):
        return self.input_ids[idx],self.token_type_ids[idx],self.attention_mask[idx]
    def load_y(self,idx:int):
        pad_id = self.vocab.get('[PAD]', 0)
        cls_id = self.vocab.get('[CLS]', 101)
        sep_id = self.vocab.get('[SEP]', 102)
        msk_token = self.vocab.get("[MASK]",103)
        # lis = self.df['input_ids'].to_list()[idx]
        # print(ast.literal_eval(f"{lis}"))
        # print(type(self.input_ids[idx]))
        x_input_ids = np.copy(self.input_ids[idx])
        
        loss_mask = np.zeros_like(self.input_ids[idx],dtype=np.float32)
        # print(len(x_input_ids))
        for i in range(256):
            token_id = x_input_ids[i]
            if token_id in (cls_id, sep_id, pad_id):
                continue
            if np.random.rand() < 0.15:
                    loss_mask[i] = 1.0 
                    chance = np.random.rand()
                    if chance < 0.8:
                        x_input_ids[i] = msk_token
                    elif chance < 0.9:
                        random_word_id = np.random.randint(104, self.vocab_size) 
                        x_input_ids[i] = random_word_id
                    else:
                        pass
        return x_input_ids,loss_mask

# loader = DataLoader(path="Bert From Scratch/model/tokenized_max_len256.parquet")
# input_ids,loss_mask = loader.load_y(1)
# print(input_ids)
# print(loss_mask)

