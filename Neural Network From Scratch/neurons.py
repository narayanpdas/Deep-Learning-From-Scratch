import numpy as np
from numpy.typing import NDArray 
import math,random,gc
from typing import Callable,List,Dict,Tuple
from activation_function import relu,d_leaky_relu
from loss_functions import CategoricalCrossEntropy as CCE


def calc_loss(true_val:NDArray,pred_val:NDArray):
    if true_val.shape != pred_val.shape:
        raise BaseException(f"Ouput Shapes Mismatach true_val shape :{true_val.shape}, pred_val shape: {pred_val.shape}")
    return CCE(true_vals=true_val,model_vals=pred_val)


class Layer():
    def __init__(self,
                units:int,
                input_size:int,
                activation_function:Callable[[float],float]=relu
        ):
        self.units = units
        self.input = input_size
        self.weights = np.random.randn(input_size,units) * np.sqrt(2.0/input_size)
        self.biases = np.zeros(units)
        self.activate = activation_function
        
    def _get_layer(self):
        return self.weights
    
    def forward_pass(self,_prev_features:NDArray)->Tuple[NDArray,NDArray]:
        z = np.dot(_prev_features,self.weights) + self.biases
        a = self.activate(z)
        return z,a
    
    def __repr__(self):
        return f"Layer(unit:{self.units} input_size = {self.input})"
    

class Model:
    def __init__(self,learning_rate:np.float32=0.01):
        self.layers:List[Layer] = []
        self.lr = learning_rate
        self.cache:Dict[str,NDArray] = {}
        self.gradients:Dict[str,NDArray] = {} 
        self._history :Dict[str,float] = {}
    def add(self,layer:Layer):
        self.layers.append(layer)
    def history(self):
        return self._history
    def forward_pass(self,X:NDArray)->NDArray:
        self.cache['A0'] = X
        A = X
        for i,layer in enumerate(self.layers):
            Z,A = layer.forward_pass(A)
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        return A
    
    def backward_pass(self,A_final:NDArray,Y_true:NDArray):
        m = Y_true.shape[0] 
        if m == 0: return
        dZ = A_final - Y_true
        num_layers = len(self.layers)
        for i in reversed(range(num_layers)):
            A_prev = self.cache[f'A{i}']
            dW = (1/m) * np.dot(A_prev.T, dZ)
            db = (1/m) * np.sum(dZ, axis=0)
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

    def train(self, X_batch: NDArray, Y_batch: NDArray):
        """Runs entire training iteration for once (Forward -> Loss -> Backward -> Update)"""
        A_final = self.forward_pass(X_batch)
        m = Y_batch.shape[0]
        if m == 0:
            return
        batch_loss = calc_loss(Y_batch, A_final)
        self.backward_pass(A_final, Y_batch)
        clip_threshold = 1.0 
        total_norm_sq = 0
        for i in range(len(self.layers)):
            total_norm_sq += np.sum(self.gradients[f'dW{i+1}']**2)
            total_norm_sq += np.sum(self.gradients[f'db{i+1}']**2)
        
        global_norm = np.sqrt(total_norm_sq)

        if global_norm > clip_threshold:
            clip_factor = clip_threshold / (global_norm + 1e-6) 
            # print(f"Clipping gradients, Norm: {global_norm:.2f}, Factor: {clip_factor:.4f}")
            for i in range(len(self.layers)):
                self.gradients[f'dW{i+1}'] *= clip_factor
                self.gradients[f'db{i+1}'] *= clip_factor
        self.update()
        return batch_loss
    def fit(self,X_train, Y_train, epochs:int=1, batch_size:int=1,verbose:int=1):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                end = i + batch_size
                X_batch = X_train[i:end]
                Y_batch = Y_train[i:end]
                batch_loss = self.train(X_batch, Y_batch)
                epoch_loss += batch_loss * X_batch.shape[0]
            avg_epoch_loss = epoch_loss / num_samples
            self._history[f'Epoch {epoch}'] = avg_epoch_loss
            if verbose:
                print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
    def predict(self,Y_test):
        return self.forward_pass(Y_test)    
    def __repr__(self):
        return f"<Model layers: {self.layers} lr: {self.lr} >"



# gc.collect()




