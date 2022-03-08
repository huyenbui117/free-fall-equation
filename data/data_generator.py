import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 9.8


def data_generate(n_sample: int, v0: float, n_error: int = 3,
                  decimals=2, seed=None):
    '''
    n_sample: số lượng dữ liệu

    v0: vận tốc ban đầu

    n_error: số lần áp dụng sai số vào mẫu

    seed: random seed

    return: vị trí theo trục y, thời gian
    '''
    np.random.seed(seed)

    time_of_fall = 2 * v0 / G

    epsilon = np.random.standard_normal(size=(n_sample, n_error))

    epsilon = epsilon / epsilon.max()

    t = np.random.uniform(low=0, high=time_of_fall, size=n_sample)

    t = t.round(decimals)

    # thêm sai số khi đo
    t_mul = t.reshape(n_sample, 1) + epsilon

    y = v0*t_mul - (G/2)*np.power(t_mul, 2)

    # lấy trung bình sai số
    y = np.mean(y, axis=-1)
    y[y < 0] = 0

    # thêm sai số dụng cụ
    y = y.round(decimals)

    return np.squeeze(y), t


def to_csv(y, t, file_path, index=False):
    df = pd.DataFrame({"t": t, "y": y})

    df.to_csv(file_path, index=index)


if __name__ == "__main__":
    y_train, t_train = data_generate(n_sample=500, v0=10, n_error=5, seed=188)
    y_test, t_test = data_generate(n_sample=200, v0=10, n_error=5, seed=202)

    # plt.plot(t_train, y_train, 'ro')
    # plt.plot(t_test, y_test, 'gx')
    # plt.show()

    to_csv(y_train, t_train, "./data/train.csv")
    to_csv(y_test, t_test, "./data/test.csv")
