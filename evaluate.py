from preparation import data_preparation
from model import PolynomailRegression
import numpy as np
import matplotlib.pyplot as plt
def evalulate() :   
     
    # Create dataset
     
    X_train, Y_train = data_preparation("./data/train.csv")
    Y_test, Y_test = data_preparation("./data/test.csv")
    # X_train = np.array( [ [1], [2], [3], [4], [5], [6], [7] ] )
     
    # Y_train = np.array( [ 45000, 50000, 60000, 80000, 110000, 150000, 200000 ] )

    # model training
     
    model = PolynomailRegression( degree = 2, learning_rate = 0.01, iterations = 10000 )
 
    model.fit( X_train, Y_train )
     
    # Prediction on training set
 
    Y_pred = model.predict( X_train )
    print(X_train.shape)
    print(Y_pred.shape)
     
    # Visualization
     
    plt.scatter( X_train, Y_train, color = 'blue' )
    
    plt.scatter( X_train, Y_pred, color = 'orange' )
     
    plt.title( 'X vs Y' )
     
    plt.xlabel( 'X' )
     
    plt.ylabel( 'Y' )
     
    plt.show()
 
 
if __name__ == "__main__" :
     
    evalulate()