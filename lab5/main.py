from enum import Enum

import numpy as np
import matplotlib.pyplot as plt


class KernelType(Enum):
    GAUSSIAN = 1
    WINDOW_RECTANGULAR = 2
    WINDOW_TRIANGULAR = 3


def vkernel(x, XN, h, kernel_type):
    N = len(XN)
    x = np.array(x)
    XN = np.array(XN)

    densities = np.zeros_like(x, dtype=float)

    for xi in XN:
        u = (x - xi) / h

        if kernel_type == KernelType.GAUSSIAN:
            K = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)

        elif kernel_type == KernelType.WINDOW_RECTANGULAR:
            K = np.where(np.abs(u) < 1, 1 / 2, 0)

        elif kernel_type == KernelType.WINDOW_TRIANGULAR:
            K = np.where(np.abs(u) < 1, 1 - np.abs(u), 0)

        densities += K

    return densities / (N * h)


NN = np.arange(1000, 30001, 1000)
err_gauss = np.zeros_like(NN, dtype=float)
err_rect = np.zeros_like(NN, dtype=float)
err_tri = np.zeros_like(NN, dtype=float)

for t, N in enumerate(NN):
    n = 1
    r = 0.5
    h = N ** (-r / n)

    x0 = np.arange(-3, 3.01, 0.05)

    XN = -np.log(np.random.rand(N))

    p_true = np.zeros_like(x0)
    mask = x0 > 0
    p_true[mask] = np.exp(-x0[mask])

    p_gauss = vkernel(x0, XN, h, KernelType.GAUSSIAN)
    p_rect = vkernel(x0, XN, h, KernelType.WINDOW_RECTANGULAR)
    p_tri = vkernel(x0, XN, h, KernelType.WINDOW_TRIANGULAR)

    err_gauss[t] = np.sqrt(np.mean((p_true - p_gauss) ** 2))
    err_rect[t] = np.sqrt(np.mean((p_true - p_rect) ** 2))
    err_tri[t] = np.sqrt(np.mean((p_true - p_tri) ** 2))

plt.figure(figsize=(10, 6))
plt.plot(NN, err_gauss, '-r', label="Гауссовское ядро")
plt.plot(NN, err_rect, '-g', label="Прямоугольное ядро")
plt.plot(NN, err_tri, '-b', label="Треугольное ядро")
plt.xlabel("Объём обучающей выборки N")
plt.ylabel("Среднеквадратичная ошибка")
plt.title("Зависимость ошибки от числа обучающих данных (Метод Парзена)")
plt.grid(True)
plt.legend()
plt.show()
