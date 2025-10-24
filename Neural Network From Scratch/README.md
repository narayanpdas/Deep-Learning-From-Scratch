# Fully-Connected Neural Network

### This project is a complete implementation of a deep neural network from scratch using only NumPy. The goal is to demonstrate a fundamental understanding of the core mechanics of a neural network, including forward propagation, backpropagation, and vectorized gradient updates.

### This code is not a "black box" framework; it is a "glass box" built to show the math and logic that powers tools like Keras and PyTorch.

### There are 2 main classes here, Layer and a Model Class:

### Layer Class: Stores the Weights and Biases of a layer, takes in the number of units/neurons per layer. Have some other functionalities like forward pass etc.

### Model Class: The main Class , housing all the hyper-parameters used in the training and holds all the layers together. Layers can be simply added one after another.Implements all the basic functions to train a neural network like forward pass , backpropagation , .fit for batch training etc.

## Experiments and Results:

![Experiments_Based_on_Model_Complexity]()

### Color Code as Shown in the curve : Green < Blue < Red < Cyan

### Order Of Model Complexity: LOW ------>--->---->--->------- HIGH

### So the Experiments Shows the Epochs and Loss Curves, Most of the time When the Experiment was Conducted it was obvserved that the Simpler Model like Green and Blue have smooth curves with lower losses compared to Others, This proves the fact that:

### _"For a simple enough problem lower and simpler models can outperform complex ones"_

![Model_Accuracy_Scores]()

### _This Represents the Accuracy Score and results for the best/simpler model, here a 80-20 split was used to compute the Results._

## How To Use:

### Check the Iris Implementation.ipynb file in the current directory, it contains a basic usecase.

## Note: The Entire System is built keeping certain things in mind such as,

- This Model would be used for a classification problem
- This Network and repo doesn't house complex activation functions or many key functionalities like an optimizer.
- For any different type of use cases like regression the Model class needs to be tweaked a bit and a carefull crafting is required while making a newer model object.
- Tune the hyperparameters carefully, could encounter a exploding Gradient problem as many advance features are not implemented.
