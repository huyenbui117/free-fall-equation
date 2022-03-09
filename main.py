import numpy as np
import pandas as pd
from model import LinearRegression, root_mean_square_error, r_squared


class make_batch:
    def __init__(self, x: np.array, batch_size: int):
        self.x = x
        self.batch_size = batch_size

    def __iter__(self):
        self.batch_idx = 0

        return self

    def __next__(self):
        start = self.batch_idx * self.batch_size
        end = min(start + self.batch_size + 1, len(self.x))

        self.batch_idx += 1

        if start > len(self.x):
            raise StopIteration
        else:
            return self.x[start:end]


def create_polynomial(x: np.array, degree):
    res = np.ones_like(x)

    for i in range(1, degree + 1):
        res = np.append(res, np.power(x, i), axis=1)

    return res


def fit(x, y, lr=1e-4, batch_size=32, epochs=3):
    model = LinearRegression(x.shape[1], biases=False)

    for epoch in range(1, epochs + 1):
        log_loss = []
        for batch_x, batch_y in zip(make_batch(x, batch_size),
                                    make_batch(y, batch_size)):
            y_pred = model.forward(batch_x)

            loss, loss_grad = root_mean_square_error(y_pred, batch_y)
            log_loss.append(loss)

            model.backward(lr, loss_grad)

        # print(f"Loss on epoch {epoch}: {sum(log_loss) / len(log_loss)}")

    return model


def validate(model, x_val, y_val):
    y_pred = model.forward(x_val)

    r2 = r_squared(y_pred, y_val)

    print("R squared score", r2)

    return r2


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    x_train = train_df.t.to_numpy().reshape(-1, 1)
    y_train = train_df.y.to_numpy().reshape(-1, 1)

    x_val = test_df.t.to_numpy().reshape(-1, 1)
    y_val = test_df.y.to_numpy().reshape(-1, 1)

    output = {"degree": [], "r-squared": [], "parameters": []}

    for degree in range(0, 11):
        x_poly_train = create_polynomial(x_train, degree)
        x_poly_val = create_polynomial(x_val, degree)

        model = fit(x_poly_train, y_train, lr=1e-2, batch_size=1, epochs=20)

        output["r-squared"].append(validate(model, x_poly_val, y_val))
        output["degree"].append(degree)
        output["parameters"].append(model.weights.reshape(-1))

    pd.DataFrame(output).to_csv("./evaluate.csv", index=False)
