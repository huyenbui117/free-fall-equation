from evaluate import evalulate
import os
import json
def extract_config(config_path):
    cwd = os.getcwd() 
    script = os.path.realpath(cwd)+"\\"+config_path
    data = json.load(open(script))

    return data
def generate_config(data):
    config = {"Experiment": [],"degree": [], "learning_rate": [], "iterations": []}

    for d in range(data["degree"]["min"], data["degree"]["max"]+1):
        for lr in data["learning_rate"]:
            for it in data["iterations"]:
                for e in data["Experiment"]:
                    config["degree"].append(d)
                    config["learning_rate"].append(lr)
                    config["iterations"].append(it)
                    config["Experiment"].append(e)
    return config
def evalulate_all(config_path):
    data = extract_config(config_path)
    config = generate_config(data)
    evalulate(config = config)

if __name__ == "__main__":
    evalulate_all("config.json")
