from pathlib import Path
import numpy as np
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
np.random.seed(42)
from neurons import Layer,Model
from loss_functions import MAE


layer1 = Layer(units=1,input_size=1)
layer2 = Layer(units=1,input_size=1)

model = Model(learning_rate=0.001,loss_function=MAE)

x = np.array([[1],[2],[3]]) # Assume for we would get square of input as output
y = np.array([[1],[4],[9]]) 
model.add(layer=layer1)
model.add(layer=layer2)

model.fit(X_train=x,Y_train=y,epochs=100)
output=model.forward_pass(x)

print(model)
print(layer1._get_layer(),layer2._get_layer())
print(model.predict(X_test=np.array([[4]])))
print(output)
