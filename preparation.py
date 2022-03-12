import os
import pandas as pd
def data_preparation(path):
    cwd = os.getcwd() 
    script = os.path.realpath(cwd)+"\\"+path
    data = pd.read_csv(script)
    X = data.iloc[:, 0].values.reshape(-1,1)
    y = data.iloc[:, 1].values
    return X,y