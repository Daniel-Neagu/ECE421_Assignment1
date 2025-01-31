import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    #Add implementation here
    #need to calculat W*, the weights that result in the least mse
    #from calculations in lecture/notes we know that w* = (XT*X)^(-1)*XT*y
    #this is only given that XT*X is invertible
    X_train = np.append(np.ones((np.shape(X_train)[0],1)),X_train,axis=1)
    w_target = np.linalg.pinv(X_train.transpose()@X_train)@X_train.transpose()@y_train

    return w_target

def mse(X_train,y_train,w):
    #Add implementation here
    #computes the mean squared error of X_train*W against the true Y_train values
    #initializes the sqr_error to 0
    X_train = np.append(np.ones((np.shape(X_train)[0],1)),X_train,axis=1)
    sqr_error = 0
    #iterates through the training values and adds up the squared error from each 
    for i,x in enumerate(X_train):
        sqr_error += (y_train[i]-(pred(x,w)))**2

    #returns the sqr_error across all the samples, divided by the number of samples to get the mse!
    return sqr_error / np.size(y_train)

def pred(X_train,w):
    #Add implementation here
    #obtains all of the y predictions by by doing X_train * W 
    #y_pred = np.dot(X_train,w)
    #multiples the arrays together to get a prediction
    y_pred = X_train @ w
    return y_pred


def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Add implementation here
    #calculates the mean squared error using scikit 
    #initializes the linear regression model
    lin_regression_model = linear_model.LinearRegression()
    #trains the model on the input training data
    lin_regression_model.fit(X_train, Y_train) 

    #predicts the output of the input testing data
    Y_pred = lin_regression_model.predict(X_test)
    #computes the MSA between the true Y_test values and the predicted Y_test values
    MSE = mean_squared_error(Y_test, Y_pred)
    return MSE


    return 0
def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)


print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()


"""


x = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
print(np.shape(x))
y = np.ones((np.shape(x)[0],1))
print(y)

print(f"w_target {fit_LinRegr(x,y)}")


c = np.append(y,x, axis=1)
print(c)"""