from layers.layer import Layer
from layers.multi_head_attention_layer import Multi_Head_Attention_Layer
from layers.feed_forward_network import FFN
from layers.normalization_layer import LayerNorm
from numpy.typing import NDArray




class Transformer_Encoder(Layer):
    def __init__(self,
                head_units:int=12,
                head_dim:int=64,
                hidden_size:int=768,
                factor:int=4):
        self.attention_layer = Multi_Head_Attention_Layer(head_units=head_units,head_dim=head_dim)
        self.norm1 = LayerNorm(hidden_units=hidden_size)
        self.feed_forward_network = FFN(hidden_size=hidden_size,factor=factor)
        self.norm2 = LayerNorm(hidden_units=hidden_size)
        
    def forward_pass(self,x:NDArray,attention_mask:NDArray):
        x_attn = self.attention_layer.forward_pass(x,attention_mask)
        x_1 = x + x_attn
        x_norm1 = self.norm1.forward_pass(x_1)
        x_ffn = self.feed_forward_network.forward_pass(x_norm1)
        x_2 = x_norm1 + x_ffn 
        x_norm2 = self.norm2.forward_pass(x_2)
        return x_norm2
    
    def backward_pass(self,d_out:NDArray):
        d_x_2 = self.norm2.backward_pass(d_out)
        d_norm2_2 = d_x_2
        d_ffn = d_x_2
        d_norm2_1 = self.feed_forward_network.backward_pass(d_ffn)
        
        d_norm2 = d_norm2_1 + d_norm2_2
        d_x_1 = self.norm1.backward_pass(d_norm2)
        
        d_x_p2 = d_x_1 
        d_attn = d_x_1
        d_x_p1 = self.attention_layer.backward_pass(d_attn)
        
        d_x = d_x_p1 + d_x_p2
        return d_x
    
    def update(self, lr):
        self.attention_layer.update(lr)
        self.norm1.update(lr)
        self.feed_forward_network.update(lr)
        self.norm2.update(lr)