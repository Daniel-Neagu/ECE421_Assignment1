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
    for epoch in range(0,max_epochs):
        #print(f"Epoch #{epoch}")
        if(errorPer(X_train,y_train,w_target)==0):
            return w_target
        else:
            #needs to find the first misclassified point and update w by it
            for i,x in enumerate(X_train):
                #print(f"Epoch #{epoch}: real value #{i}: {y_train[i]} prediction: {pred(x,w)}")
                if(pred(x,w)!=y_train[i]):
                    #print(f"Epoch #{epoch}: misclassified point at: Index [{i}]")
                    #print(f"Epoch #{epoch}: Updating w from {w} to {w + y_train[i]*x}")
                    w = w + y_train[i]*x
                    break

            if(errorPer(X_train,y_train,w_target)>errorPer(X_train,y_train,w)):
                print(f"Epoch #{epoch} improved w* from {w_target} to {w}, error: {errorPer(X_train,y_train,w_target)} to {errorPer(X_train,y_train,w)}")
                w_target = w

            #need to find a way to make this stopin the case that the data is not lineearly seperable, 
            #additionally, i need to figure out how tokeep track of the current best w_target so far and 
            #keep it stored bc not every update may be beneficial 

    print(f"training done: average error: {errorPer(X_train,y_train,w_target)}")
    return w_target


def errorPer(X_train,y_train,w):
    #Add implementation here 
    #outputs the avg number of error points output by w!
    numerrors =0
    for i,x in enumerate(X_train):
        if(pred(x,w)!=y_train[i]):
            numerrors+=1
    avgerrors = numerrors / np.size(y_train)

    return avgerrors
    

def confMatrix(X_train,y_train,w):
    #Add implementation here 
    outputMatrix = np.zeros((2,2))
    for i,x in enumerate(X_train):
        index1 = int((y_train[i]+1)/2)
        index2 = int((np.sign(np.matmul(w.transpose(),x))+1)/2)
        outputMatrix[index1][index2] +=1
    return outputMatrix

def pred(X_i,w):
    #Add implementation here
    #outputs the output of the perceptron models prediction of X_i's label based on w
    return np.sign(np.matmul(w.transpose(),X_i))


def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here
    return 0 

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
    

test_Part1()

"""
if __name__ == "__main__":
    
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1
    fit_perceptron(X_train,y_train)
"""

