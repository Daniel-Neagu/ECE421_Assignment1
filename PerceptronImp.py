import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 
import random

def fit_perceptron(X_train, y_train):
    #Add implementation here 
    #initializes the set w to an array of 0s to the same dimensions as an input
    w = np.zeros(np.shape(X_train[0]))
    w_target = np.array(w)
    max_epochs = 5000
    for i in range(0,max_epochs):
        if(errorPer(X_train,y_train,w)==0):
            return w_target
        else:
            return w_target

    return w_target


def errorPer(X_train,y_train,w):
    #Add implementation here 
    #outputs the avg number of error points output by w!
    return 0
    

def confMatrix(X_train,y_train,w):
    #Add implementation here 
    numerrors =0
    for x,i in enumerate(X_train):
        if(pred(x,w)!=y_train[i]):
            numerrors+=1

    print(numerrors)
    print(np.size(y_train)[0])
    avgerrors = numerrors / np.size[y_train][0]
    print(avgerrors)
    return 

def pred(X_i,w):
    #Add implementation here
    #outputs the output of the perceptron models prediction of X_i's label based on w
    return np.sign(np.matmul(w.transpose(),X_i))

"""
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here 

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()"""


if __name__ == "__main__":
    """
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    fit_perceptron(X_train,y_train)
    """
    a = np.array([1,2.2,3,4])
    b = np.array([2,1.1,-10,1])
    print(a)
    print(b)
    print(pred(a,b))

