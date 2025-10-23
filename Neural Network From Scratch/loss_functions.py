import numpy as np
from numpy.typing import NDArray
import math

# Mean Squarred Error

def MSE(true_vals:NDArray,model_vals:NDArray)->float:
    if len(true_vals) != len(model_vals):
        raise ValueError("Error: Arrays Not the same Size!")
    else:
        _n = len(true_vals)
        _func = lambda t_val,m_val: t_val**2 + m_val**2 - 2*t_val*m_val
        _mse = sum([_func(t_val,m_val)for t_val,m_val in zip(true_vals,model_vals)]) / _n
        return _mse

# Mean Absolute Error

def MAE(true_vals:NDArray,model_vals:NDArray)->float:
    if len(true_vals) != len(model_vals):
        raise ValueError("Error: Arrays Not the same Size!")
    else:
        _n = len(true_vals)
        _func = lambda t_val,m_val: abs(t_val - m_val)
        _mse = sum([_func(t_val,m_val)for t_val,m_val in zip(true_vals,model_vals)]) / _n
        return _mse


# y_true = np.array([1,2,3,4])
# y_pred = np.array([2,3,1,5])
# print(f'For Regression :')
# print(f'True Values = {y_true}',f'Model Values = {y_pred}')
# print(f'Mean Squared Error Loss: {MSE(y_true,y_pred)}')
# print(f'Mean Absolute Error Loss: {MAE(y_true,y_pred)}')

# For Binary Classification
# Assumed that we are trying to predict 1 with the Model.
# So the y_pred is the probability of getting the class 1 ,where class 0 is 0 and class 1 is 1

# Binary Cross-Entropy 
def BCE(true_vals:NDArray,model_vals:NDArray)->float:
    if len(true_vals) != len(model_vals):
        raise ValueError("Error: Arrays Not the same Size!")
    else:
        _func = lambda t_val,m_val: (t_val*math.log(m_val) + (1-t_val)*math.log(1-m_val))
        _bce = -sum([_func(t_val,m_val)for t_val,m_val in zip(true_vals,model_vals)])
        return _bce
    

# Binary Classification Data -> multiple/batch example
# y_true = np.array([0,1,0,1,1])
# y_pred = np.array([0.9,0.7,0.1,0.3,0.81]) # Probabiltiy if class is 1 or not
# y_pred_perfect = np.array([0.9,0.1,0.1,0.9,0.91]) # Checking for loss function working


# print("For Binary Classification :")
# print(f'True Values = {y_true}',f'Model Values = {y_pred}')
# print(f'Binary Cross-Entropy Loss : {BCE(y_true,y_pred)}')
# print(f'Binary Cross-Entropy Loss near perfect : {BCE(y_true,y_pred_perfect)}')


# For Multi Class Classification
# Categorical Cross Entropy
def CategoricalCrossEntropy(true_vals:NDArray,model_vals:NDArray)->float:
    if len(true_vals) != len(model_vals):
        raise ValueError("Error: Arrays Not the same Size!")
    else:
        model_vals_clipped = np.clip(model_vals, 1e-9, 1.0)
        return -np.sum(true_vals * np.log(model_vals_clipped))/true_vals.shape[0] 
    
# Categorical Classification Data -> For a single example, One Hot encoded data is used
# y_true = np.array([0,1,0,0,0])
# y_pred = np.array([0.1,0.4,0.1,0.3,0.1]) # Probability for Corresponding Classes
# y_pred_perfect = np.array([0,1,0,0,0]) 

# print("For Multi Class Classification :")
# print(f'True Values = {y_true}',f'Model Values = {y_pred}')
# print(f'Categorical Cross-Entropy Loss : {CategoricalCrossEntropy(y_true,y_pred)}')
# print(f'Categorical Cross-Entropy Loss near perfect : {CategoricalCrossEntropy(y_true,y_pred_perfect)}')
