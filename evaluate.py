from preparation import data_preparation
from model import PolynomailRegression, visualize, visualize_show


def evalulate(train=True, validate=True, test=True, config=None):
    X_train, Y_train = data_preparation("data\\train.csv")
    X_test, Y_test = data_preparation("data\\test.csv")

    if config == None:
        config = {"degree": [2], "learning_rate": [0.03], "iterations": [1000]}
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
            model.save_model(model_name)

        # load model
        if validate:
            path = "savemodel\\" + model_name + ".pkl"
            model = model.load_model(path)

            # Prediction on test set

            Y_pred = model.predict(X_test)
        print(model.get_parameters())
        # output

        # Visualize
        visualize(X_test, Y_test, "blue")
        visualize(X_test, Y_pred, "orange")
        visualize_show(model_name)
    return None

if __name__ == "__main__":
    evalulate()
