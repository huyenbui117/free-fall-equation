import pandas as pd
import os
def data_preparation(path):
    cwd = os.getcwd() 
    script = os.path.realpath(cwd)+path
    data = pd.read_csv("D:/free-fall-equation/data/train.csv")
    X = data.iloc[:, 0].values.reshape(-1,1)
    y = data.iloc[:, 1].values
    return X,y
