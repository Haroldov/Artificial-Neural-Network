import numpy as np
from trainNN import trainNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from classify import classify
from drawNN import draw_neural_net
import matplotlib.pyplot as plt
with open('datos.tsv', 'r+') as f:
    text = f.read()
    f.seek(0)
    f.truncate()
    f.write(text.replace(',', '.'))
tmp = np.genfromtxt("datos.tsv",delimiter="\t", skip_header=0, filling_values=1,dtype=float)
y = tmp[:,-1]
y[y == -1] = 2

X_train, X_test, Y_train, Y_test = train_test_split(tmp[:,0:-1], y, test_size=0.3)

Input_layer_size = np.size(X_train,axis=1)
First_hidden_layer_size = 10
Second_hidden_layer_size = 10
num_labels = 2

W  = trainNN( X_train, Y_train, Input_layer_size ,First_hidden_layer_size, Second_hidden_layer_size, num_labels, alpha = 1)

testClass = classify(W,Input_layer_size ,First_hidden_layer_size, Second_hidden_layer_size, num_labels,X_test)

print(testClass,Y_test)
print("\nLa matriz de confusión es: \n\n",confusion_matrix(Y_test,testClass))
W2 = np.reshape(W[0], (First_hidden_layer_size, Input_layer_size + 1), order='F')
W3 = np.reshape(W[1], (Second_hidden_layer_size, First_hidden_layer_size + 1), order='F')
W4 = np.reshape(W[2], (num_labels, Second_hidden_layer_size + 1), order='F')
print("\nLos pesos de la primera capa oculta son: \n\n", W2)
print("\nLos pesos de la segunda capa oculta son: \n\n", W3)
print("\nLos pesos de la capa de salida son: \n\n", W4)
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [2, 10, 10, 2])
plt.show()
