import numpy as np
import numpy.matlib as mtl
def sigmoid(X):
    out = 1/(1+np.exp(-1*X))
    return out

def sigmoidGradient(X):
    out = sigmoid(X)*(1-sigmoid(X))
    return out

def costFunction(nnParams,Input_layer_size ,First_hidden_layer_size, Second_hidden_layer_size, num_labels, X, Y):

    W2 = np.reshape(nnParams[0],(First_hidden_layer_size,Input_layer_size+1),order = 'F').T
    W3 = np.reshape(nnParams[1],(Second_hidden_layer_size,First_hidden_layer_size+1),order = 'F').T
    W4 = np.reshape(nnParams[2],(num_labels,Second_hidden_layer_size+1),order='F').T


    #Number of samples
    m = np.size(X,axis=0)

    #Feed Forward
    X = np.concatenate((np.ones((m,1)),X),axis = 1)
    O1 = sigmoid(W2.T@X.T)
    O1 = np.concatenate((np.ones((np.size(O1.T,axis=0), 1)), O1.T), axis=1)
    O2 = sigmoid(W3.T@O1.T)
    O2 = np.concatenate((np.ones((np.size(O2.T,axis=0), 1)), O2.T), axis=1)
    h = sigmoid(W4.T@O2.T).T

    Y = mtl.repmat(Y, num_labels, 1).T
    Y = Y/mtl.repmat(np.arange(1,num_labels+1),m,1)
    Y[Y == 1] = 1
    Y[Y != 1] = 0
    J = ((-1/m) * ((Y.T@np.log(h)) + ((1-Y.T)@np.log(1-h))))
    J = np.sum(np.diag(J))

    #Backpropagation Algorithm
    d_4 = h-Y
    d_3 = (W4[1:,:]@d_4.T)*sigmoidGradient(W3.T@O1.T)
    d_2 = (W3[1:,:]@d_3)*sigmoidGradient(W2.T@X.T)
    A_4 = d_4.T@O2
    A_3 = d_3@O1
    A_2 = d_2@X
    weight4_grad = A_4/m
    weight3_grad = A_3/m
    weight2_grad = A_2/m


    grad = np.array([weight2_grad.reshape(weight2_grad.size,order='F'), weight3_grad.reshape(weight3_grad.size,order='F')\
                     ,weight4_grad.reshape(weight4_grad.size,order='F')])

    return(J,grad)

print()