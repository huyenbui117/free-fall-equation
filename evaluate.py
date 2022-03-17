from preparation import data_preparation
from model import PolynomailRegression
from utils import visualize, visualize_save, r2_metrics
import pandas as pd
import os

def evalulate(train=True, validate=True, test=True, config=None):
    X_train, Y_train = data_preparation("data\\train.csv")
    X_test, Y_test = data_preparation("data\\test.csv")

    
    if config == None:
        config = {"Experiment": ["PolynomialRegression"],"degree": [2], "learning_rate": [0.03], "iterations": [1000]}
    result =  {"Experiment":[],  "model":[], "metrics":[]}
    # Create dataset
    for i, _ in enumerate(config["degree"]):
        model = PolynomailRegression(degree=config["degree"][i],
                                     learning_rate=config["learning_rate"][i],
                                     iterations=config["iterations"][i])
        model_name = "d" + str(config["degree"][i]) + "_lr" + str(config["learning_rate"][i])\
                     + "_iter" + str(config["iterations"][i])
        # model training
        if train:
            model.fit(X_train, Y_train)
            model.save_model(model_name, config["Experiment"][i])

        # load model
        if validate:
            path = "savemodel\\" + str(config["Experiment"][i])+ "\\" +model_name+"\\" +model_name + ".pkl"
            model = model.load_model(path)

            # Prediction on test set

            Y_pred = model.predict(X_test)
            result["Experiment"].append(config["Experiment"][i])
            result["model"].append(model_name)
            result["metrics"].append(r2_metrics(Y_pred, Y_test))
        print(model.get_parameters())
        # output
        cwd = os.getcwd() 
        script = os.path.realpath(cwd)+"\\"+"result\\"
        if not os.path.exists(script):
            os.makedirs(script)
        path = script + "result.csv"    
        result_df = pd.DataFrame(result)
        print(result_df)
        print(path)
        result_df.to_csv(path)
        # Visualize
        visualize(X_test, Y_test, "blue", new_plot=True)
        visualize(X_test, Y_pred, "orange")
        visualize_save(model_name, config["Experiment"][i])
    return None

if __name__ == "__main__":
    evalulate()
