import numpy as np
from model import root_mean_square_error


def check_grad(func, x, grad, epsilon=1e-6):
    x_upper = x + epsilon
    x_lower = x - epsilon

    true_grad = (func(x_upper) - func(x_lower)) / (2 * epsilon)

    true_grad.shape == grad.shape

    true_grad = np.mean(true_grad)
    grad = np.mean(grad)

    err = abs(true_grad - grad)
    return err <= 2 * epsilon


if __name__ == "__main__":
    x = np.random.rand(5, 1)
    y = np.random.rand(5, 1)

    assert check_grad(lambda _x: root_mean_square_error(_x, y)[0],
                      x, root_mean_square_error(x, y)[1])

    print("Passed!")
