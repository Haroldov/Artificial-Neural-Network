import numpy as np
from costFunction import costFunction
def trainNN( X, Y, Input_layer_size ,First_hidden_layer_size, Second_hidden_layer_size, num_labels, alpha):

    #Initialize random weights
    W2 = np.random.rand(First_hidden_layer_size, Input_layer_size+1)*2*10e-1 - 10e-1
    W3 = np.random.rand(Second_hidden_layer_size, First_hidden_layer_size+1)*2*10e-1 - 10e-1
    W4 = np.random.rand(num_labels, Second_hidden_layer_size+1)*2*10e-1 - 10e-1

    #Begin gradient descent
    i = 1 # Number of iterations
    cost = 10e4 #cost to compare
    while True:
        # Unroll parameters to calculate gradients and cost
        nnParams = np.array([W2.reshape(W2.size, order='F'), W3.reshape(W3.size, order='F')\
                                , W4.reshape(W4.size, order='F')])
        #Compute cost and gradients
        [J, grad] = costFunction(nnParams, Input_layer_size, First_hidden_layer_size, Second_hidden_layer_size,\
                     num_labels, X, Y)


        #Print some things for control
        print('Iteración: ',i,'| Costo: ',J)

        #Check cost
        if J < 0.1 :
            break
        if abs(J-cost) < 10e-20 :
            break

        #Update cost to stop program
        cost = J

        #Update weights

        W2_grad = np.reshape(grad[0], (First_hidden_layer_size, Input_layer_size + 1),order='F')
        W3_grad = np.reshape(grad[1], (Second_hidden_layer_size, First_hidden_layer_size + 1),order='F')
        W4_grad = np.reshape(grad[2], (num_labels, Second_hidden_layer_size + 1),order='F')

        W2 = W2 - alpha * W2_grad
        W3 = W3 - alpha * W3_grad
        W4 = W4 - alpha * W4_grad

        #Next iteration
        i = i+1

    #Return final weights
    nnParams = np.array([W2.reshape(W2.size, order='F'), W3.reshape(W3.size, order='F')\
                            , W4.reshape(W4.size, order='F')])

    return(nnParams)




