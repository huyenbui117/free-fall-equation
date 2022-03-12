from preparation import data_preparation
from model import PolynomailRegression, visualize, visualize_show
def evalulate(config_path=None) :   
     
    # Create dataset
     
    X_train, Y_train = data_preparation("data\\train.csv")
    X_test, Y_test = data_preparation("data\\test.csv")

     
    model = PolynomailRegression( degree = 2, learning_rate = 0.03, iterations = 10000 )
    
    # model training
    
    model.fit( X_train, Y_train )
    model.save_model("model")

    #load model

    model = model.load_model("savemodel\\model.pkl")
    
    # Prediction on test set
 
    Y_pred = model.predict( X_test )
    print(model.get_parameters())
    #output
    
    #Visualize
    visualize(X_test, Y_test, "blue")
    visualize(X_test, Y_pred, "orange")
    visualize_show()
 
if __name__ == "__main__" :
     
    evalulate()