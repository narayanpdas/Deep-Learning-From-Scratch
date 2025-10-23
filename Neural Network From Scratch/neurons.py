import numpy as np
from numpy.typing import NDArray 
import math,random,gc
from typing import Callable,List,Dict
from activation_function import leaky_relu,relu,softmax,d_leaky_relu
from loss_functions import CategoricalCrossEntropy as CCE


def calc_loss(true_val:NDArray,pred_val:NDArray):
    if true_val.shape != pred_val.shape:
        raise BaseException("Ouput Shapes Mismatach")
    return CCE(true_vals=true_val,model_vals=pred_val)
class Layer():
    def __init__(self,
                units:int,
                input_size:int,
                activation_function:Callable[[float],float]=relu
        ):
        self.units = units
        self.input = input_size
        self.weights = np.random.randn(input_size,units) * 0.01
        self.biases = np.random.randn(units)*0.01
        self.activate = activation_function
        
    def _get_layer(self):
        return self.weights
    
    def forward_pass(self,_prev_features:NDArray)->list[NDArray,float]:
        z = np.dot(_prev_features,self.weights) + self.biases
        return z,self.activate(z)
    
    def __repr__(self):
        return f"Layer(unit:{self.units} input_size = {self.input})"
    

class Model:
    def __init__(self,learning_rate:np.float32=0.01):
        self.layers:List[Layer] = []
        self.lr = learning_rate
        self.cache:Dict[str,NDArray] = {}
        self.gradients:Dict[str,NDArray] = {} 
        
    def add(self,layer:Layer):
        self.layers.append(layer)
        
    def forward_pass(self,X:NDArray)->NDArray:
        self.cache['A0'] = X
        A = X
        for i,layer in enumerate(self.layers):
            Z,A = layer.forward_pass(A)
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        return A
    
    def backward_pass(self,A_final:NDArray,Y_true:NDArray):
        dZ = A_final - Y_true
        num_layers = len(self.layers)
        for i in reversed(range(num_layers)):
            A_prev = self.cache[f'A{i}']
            dW = np.dot(A_prev.reshape(-1,1),dZ.reshape(1,-1))
            db = dZ
            self.gradients[f'dW{i+1}'] = dW
            self.gradients[f'db{i+1}'] = db
            if i>0:
                dA_prev = np.dot(dZ,self.layers[i].weights.T)
                Z_prev = self.cache[f'Z{i}']
                dZ = dA_prev * d_leaky_relu(Z_prev)
    
    def update(self):
        for i,layer in enumerate(self.layers):
            layer.weights -= self.lr * self.gradients[f'dW{i+1}']
            layer.biases  -= self.lr * self.gradients[f'db{i+1}']

    def train(self, X_sample: NDArray, Y_sample: NDArray):
        """Runs entire training iteration for once (Forward -> Loss -> Backward -> Update)"""
        A_final = self.forward_pass(X_sample)
        loss = calc_loss(Y_sample, A_final)
        self.backward_pass(A_final, Y_sample)
        self.update()
        return loss
    
    def __repr__(self):
        return f"<Model layers: {self.layers} lr: {self.lr} >"


inputs = np.arange(1,6) 
true_labels = np.array([0,0,1])

model = Model(learning_rate=0.1)

model.add(layer=Layer(units=4,input_size=inputs.size,activation_function=leaky_relu))
model.add(layer=Layer(units=4,input_size=4,activation_function=leaky_relu))
model.add(layer=Layer(units=3,input_size=4,activation_function=softmax))

prev_loss = (calc_loss(true_labels,model.forward_pass(inputs)))
print(f'Loss Before training :{prev_loss}\n')

EPOCHS = 5
for i in range(0,EPOCHS):
    print(f"Running {i}th epoch: \n")
    loss_after = model.train(X_sample=inputs,Y_sample=true_labels)
    print(f'Loss After training :{loss_after}\n')

gc.collect()




