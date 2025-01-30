import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 
import random

def fit_perceptron(X_train, y_train):
    #Add implementation here 
    #initializes the set w to an array of 0s to the same dimensions as an input
    w = np.zeros(np.shape(X_train[0]))
    #initializes the best current version of the weights so far
    w_target = np.array(w)
    
    #sets the max number of epochs to 5000
    max_epochs = 5000
    #trains the model
    for epoch in range(0,max_epochs):
        #checks for if we've achieved 0 error
        if(errorPer(X_train,y_train,w_target)==0):
            return w_target
        #checks for the first first misclassified point and update w by it
        else:
            for i,x in enumerate(X_train):
                #updates w by the first misclassified point it sees
                if(pred(x,w)!=y_train[i]):
                    #print(f"Epoch #{epoch}: misclassified point at: Index [{i}]")
                    #print(f"Epoch #{epoch}: Updating w from {w} to {w + y_train[i]*x}")
                    w = w + y_train[i]*x
                    break
            #if the update made the model more accurate, we take the updated value of w for our current best fit of w_target
            if(errorPer(X_train,y_train,w_target)>errorPer(X_train,y_train,w)):
                print(f"Epoch #{epoch} improved w* from {w_target} to {w}, error: {errorPer(X_train,y_train,w_target)} to {errorPer(X_train,y_train,w)}")
                w_target = w

    #returns the best fit after 5000 iterations of calling PLA and updating our best weights
    print(f"training done: average error: {errorPer(X_train,y_train,w_target)}")
    return w_target


def errorPer(X_train,y_train,w):
    #Add implementation here 
    #outputs the avg number of error points output by wT*x!
    
    #iterates through all the data points
    #and checks whether or not our weights correctly classifh them all
    numerrors =0
    for i,x in enumerate(X_train):
        if(pred(x,w)!=y_train[i]):
            numerrors+=1
    #computs the average number of errors
    avgerrors = numerrors / np.size(y_train)

    #returns the avg errors
    return avgerrors
    

def confMatrix(X_train,y_train,w):
    #Add implementation here 
    #creates and outputs the confMatrix
    outputMatrix = np.zeros((2,2))
    #checks through the data points for which points were false/true positive/negatives
    #and adds them to the output confMatrix outputMatrix respectively
    for i,x in enumerate(X_train):
        index1 = int((y_train[i]+1)/2)
        index2 = int((np.sign(np.matmul(w.transpose(),x))+1)/2)
        outputMatrix[index1][index2] +=1

    #outputs the confMatrix
    return outputMatrix

def pred(X_i,w):
    #Add implementation here
    #outputs the output of the perceptron models prediction of X_i's label based on w
    return np.sign(np.matmul(w.transpose(),X_i))


def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here

    #creating the perceptron model, ensuring it iterates 5000 times
    perceptron = Perceptron(n_iter_no_change=5000,max_iter=5000,tol=1e-3, random_state=0)

    #fits the model
    perceptron.fit(X_train, Y_train)

    #obtains the model's prediction on the test set
    Y_pred = perceptron.predict(X_test)
    
    #returns a 2by2 confusion matrix of integer values
    confMatrix = confusion_matrix(Y_test,Y_pred)
    return confMatrix

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

