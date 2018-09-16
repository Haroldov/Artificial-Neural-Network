import numpy as np
from costFunction import sigmoid
def classify(nnParams,Input_layer_size ,First_hidden_layer_size, Second_hidden_layer_size, num_labels,X):
    W2 = np.reshape(nnParams[0], (First_hidden_layer_size, Input_layer_size + 1), order='F').T
    W3 = np.reshape(nnParams[1], (Second_hidden_layer_size, First_hidden_layer_size + 1), order='F').T
    W4 = np.reshape(nnParams[2], (num_labels, Second_hidden_layer_size + 1), order='F').T

    # Number of samples
    m = np.size(X, axis=0)

    # Feed Forward
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    O1 = sigmoid(W2.T @ X.T)
    O1 = np.concatenate((np.ones((np.size(O1.T, axis=0), 1)), O1.T), axis=1)
    O2 = sigmoid(W3.T @ O1.T)
    O2 = np.concatenate((np.ones((np.size(O2.T, axis=0), 1)), O2.T), axis=1)
    h = sigmoid(W4.T @ O2.T).T
    testClass = np.argmax(h,axis=1)+1

    return(testClass)