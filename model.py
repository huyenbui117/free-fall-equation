import numpy as np


class LinearRegression:
    def __init__(self, n_features: int, biases=True):
        self.biases = biases

        if biases:
            n_features += 1

        self.weights = np.random.rand(n_features, 1)

        self.grad = 0

    def forward(self, x: np.array):
        batch_size = x.shape[0]

        if self.biases:
            x = np.append(np.ones(batch_size, 1), x, axis=1)

        y_pred = np.matmul(x, self.weights)

        self.grad = np.transpose(x)

        return y_pred

    def backward(self, lr, loss_grad):
        weight_grad = np.matmul(self.grad, loss_grad)

        self.weights = self.weights - lr * weight_grad


def root_mean_square_error(y_pred, y):
    assert y_pred.shape == y.shape

    n = y.shape[0]

    error = y - y_pred
    mse = np.power(error, 2) / n
    rmse = np.sqrt(mse)

    loss = np.sum(rmse)

    grad = -error / (n * rmse)

    return loss, grad


def r_squared(y_pred, y):
    assert y_pred.shape == y.shape

    mu_y = np.mean(y)

    rss = np.power(y - y_pred, 2)
    tss = np.power(y - mu_y, 2)

    rs = 1 - np.sum(rss) / np.sum(tss)

    return rs
