
from layers.embedding_layer import EmbeddingLayer
from layers.normalization_layer import LayerNorm
from layers.transformer_encoder import Transformer_Encoder
from layers.activation.activation_functions import Linear,Softmax_WD
from layers.dense import Dense
from dataloader import DataLoader
from numpy.typing import NDArray
import numpy as np
from tqdm import tqdm


def CategoricalCrossEntropy(true_vals:NDArray,model_vals:NDArray)->float:
    if len(true_vals) != len(model_vals):
        raise ValueError("Error: Arrays Not the same Size!")
    else:
        model_vals_clipped = np.clip(model_vals, 1e-9, 1.0)
        return -np.sum(true_vals * np.log(model_vals_clipped), axis=-1)

MAX_TOKEN_LEN = 256
VOCAB_SIZE = 30000
HIDDEN_SIZE = 768
HEAD_DIM = 64
VOCAB_PATH = "Bert From Scratch/model/tokenizer/vocab.txt"

class BertModel():
    def __init__(self,num_blocks:int=12):
        self.embedding = EmbeddingLayer(max_token_len=MAX_TOKEN_LEN,
                                        vocab_size=VOCAB_SIZE,
                                        hidden_units=HIDDEN_SIZE)
        self.encoder_blocks = [Transformer_Encoder(
                                        head_dim=HEAD_DIM,
                                        head_units=num_blocks) for _ in range(num_blocks)]
        self.mlm_head = Dense(units=VOCAB_SIZE,
                            input_size=HIDDEN_SIZE,
                            activation_function=Linear())
        self.softmax = Softmax_WD()
        self.history:dict = {
            "epoch":[],
            "loss":[]
        }
    def forward_pass(self, input_ids, token_type_ids, attention_mask):
        x = self.embedding.forward_pass(input_ids=input_ids,token_type_ids=token_type_ids)
        for block in self.encoder_blocks:
            x = block.forward_pass(x, attention_mask)
        output_logits = self.mlm_head.forward_pass(x)
        return output_logits
    
    def backward_pass(self, d_out):
        d_x = self.mlm_head.backward_pass(d_out=d_out)
        for block in reversed(self.encoder_blocks):
            d_x = block.backward_pass(d_out=d_x)
        self.embedding.backward_pass(d_out=d_x)
        return None
    
    def update(self,lr):
        self.embedding.update(lr=lr)
        for block in self.encoder_blocks:
            block.update(lr=lr)
        self.mlm_head.update(lr=lr)
        
    def loss_func(self,y_true:NDArray,logits:NDArray,loss_mask:NDArray):
        probs = self.softmax.activate(logits)
        y_true_ohe = np.eye(VOCAB_SIZE)[y_true]
        raw_loss = CategoricalCrossEntropy(y_true_ohe,probs)
        masked_loss = raw_loss * loss_mask
        error = np.sum(masked_loss) / np.sum(loss_mask)
        
        d_logits = probs - y_true_ohe
        d_logits = d_logits * loss_mask.reshape(-1,1)
        
        return error , d_logits
    
    def fit(self,data:DataLoader):
        with tqdm(total=data.__len__(),desc="Training Bert: ",unit="batch")as pbar:
            for idx in range(data.__len__()):
                y_input_ids,token_type_ids,attention_mask = data.load_x(idx)
                x_input_ids,loss_mask = data.load_y(idx)
                logits = self.forward_pass(input_ids=x_input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
                
                loss,d_logits = self.loss_func(y_true=y_input_ids,
                                            logits=logits,
                                            loss_mask=loss_mask)
                # print(f"Loss for {idx} datapoint : {loss}")
                self.history['epoch'].append(idx)
                self.history['loss'].append(loss)
                self.backward_pass(d_out=d_logits)
                pbar.set_postfix(loss=f"{loss:.3f}")
                pbar.update(1)
    def predict(self,x_test):
        preds = []
        for x in x_test:
            self.forward_pass(x)

