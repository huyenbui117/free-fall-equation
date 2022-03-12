import numpy as np
import pickle
import os 
import matplotlib.pyplot as plt
class PolynomailRegression():
    """Base class for Linear Models"""

    def __init__(self, degree, learning_rate, iterations):
        self.degree = degree 
        self.learning_rate = learning_rate
        self.iterations = iterations
    def transform(self, X):
        X_transform = np.ones((self.row, 1)) 
        d=0
        for d in range(self.degree + 1):
            if d!=0:
                x_pow = np.power( X, d )
                 # append x_pow to X_transform
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )
        return X_transform

    # function to normalize X_transform
    def normalize( self, X ) :         
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
         
        return X
    def loss_function(pred, actual):
        return pred - actual;
    def fit( self, X, Y ) :
         
        self.X = X
     
        self.Y = Y
     
        self.row = self.X.shape[0]
     
        # weight initialization
     
        self.W = np.zeros(self.degree + 1)
         
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( self.X )
         
        # normalize X_transform
         
        X_normalize = self.normalize( X_transform )
                 
        # gradient descent learning
     
        for i in range(self.iterations) :
             
            h = self.predict( self.X )
         
            error = loss_function(h, self.Y)
             
            # update weights
         
            self.W = self.W - self.learning_rate * ( 1 / self.row ) * np.dot( X_normalize.T, error )
        return self
     
    # predict
     
    def predict( self, X ) :
        
     
        self.row,_= X.shape
      
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( X )
         
        X_normalize = self.normalize( X_transform )
        Y_pred =np.dot( X_transform, self.W )
         
        return Y_pred
    def save_model(self,model_name):
        cwd = os.getcwd() 
        script = os.path.realpath(cwd)+"\\"+"savemodel"
        if not os.path.exists(script):
            os.makedirs(script)
        script= script + "\\"+model_name+".pkl"
        pickle.dump(self, open(script, 'wb'))
    def load_model(self, path):
        cwd = os.getcwd() 
        script = os.path.realpath(cwd)+"\\"+path
        pickled_model = pickle.load(open(script,"rb"))
        return pickled_model
    def get_parameters(self):
        return self.W
def visualize( X, Y, color, new_plot=False, figure=None):
    if new_plot:
        fig = plt.figure()
    else:
        fig = figure
    plt.scatter( X, Y, color = color )
    
    plt.title( 'X vs Y' )
    
    plt.xlabel( 'X' )
    
    plt.ylabel( 'Y' )
    return fig
def visualize_show():
    plt.show()
