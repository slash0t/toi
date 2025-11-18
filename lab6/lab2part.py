import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Математические ожидания классов
m1 = np.array([2, 2])
m2 = np.array([1, -1])

# Общая матрица ковариации
C = np.array([[5, 1],
              [1, 5]])

# Априорные вероятности (предполагаем равные)
P1 = 0.5
P2 = 0.5

# Теоретический анализ
# Расстояние Махаланобиса между центрами классов
C_inv = np.linalg.inv(C)
delta_m = m1 - m2
d_mahal = np.sqrt(delta_m.T @ C_inv @ delta_m)

# Вычисляем параметры решающей функции
w = C_inv @ (m1 - m2)  # вектор весов
w0 = -0.5 * (m1.T @ C_inv @ m1 - m2.T @ C_inv @ m2) + np.log(P2/P1)

# Проекция на направление различения классов
# Стандартное отклонение проекции
sigma_proj = np.sqrt((m1 - m2).T @ C @ (m1 - m2) / (d_mahal**2))

# Теоретическая вероятность ошибки
P_error_theory = norm.cdf(-d_mahal / 2)

# Матрица ошибок (confusion matrix) - теоретическая
# P(error | Class 1) = P(classify as Class 2 | true Class 1)
P_error_given_1 = norm.cdf(-d_mahal / 2)
P_error_given_2 = norm.cdf(-d_mahal / 2)

confusion_theory = np.array([
    [1 - P_error_given_1, P_error_given_1],  # True Class 1
    [P_error_given_2, 1 - P_error_given_2]   # True Class 2
])

# Генерация обучающей выборки
np.random.seed(345621233)
n_samples = 1000

# Генерация выборок из каждого класса
X1 = np.random.multivariate_normal(m1, C, n_samples)
X2 = np.random.multivariate_normal(m2, C, n_samples)

# Классификация на основе байесовского решающего правила
def classify(x, m1, m2, C_inv, P1, P2):
    # Дискриминантная функция
    g = (m1 - m2).T @ C_inv @ x - 0.5 * (m1.T @ C_inv @ m1 - m2.T @ C_inv @ m2) + np.log(P1/P2)
    return 1 if g > 0 else 2

# Классификация образцов
y1_pred = np.array([classify(x, m1, m2, C_inv, P1, P2) for x in X1])
y2_pred = np.array([classify(x, m1, m2, C_inv, P1, P2) for x in X2])

# Вычисление экспериментальной матрицы ошибок
correct_1 = np.sum(y1_pred == 1) / n_samples
error_1 = np.sum(y1_pred == 2) / n_samples
error_2 = np.sum(y2_pred == 1) / n_samples
correct_2 = np.sum(y2_pred == 2) / n_samples

confusion_exp = np.array([
    [correct_1, error_1],
    [error_2, correct_2]
])

# Средняя экспериментальная вероятность ошибки (равные priors и одинаковый размер выборок)
P_error_exp = 0.5 * (error_1 + error_2)

# Визуализация

# Создание сетки для визуализации решающей границы
x_min, x_max = min(X1[:, 0].min(), X2[:, 0].min()) - 2, max(X1[:, 0].max(), X2[:, 0].max()) + 2
y_min, y_max = min(X1[:, 1].min(), X2[:, 1].min()) - 2, max(X1[:, 1].max(), X2[:, 1].max()) + 2

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Классификация каждой точки сетки
Z = np.array([classify(np.array([x, y]), m1, m2, C_inv, P1, P2)
              for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 14))

ax = axes[0]
im = ax.imshow(confusion_theory, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_yticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title(f'Теоретическая матрица ошибок\nОшибка = {P_error_theory:.4f}',
             fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_theory[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1]
im = ax.imshow(confusion_exp, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_yticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title(f'Экспериментальная матрица ошибок\nОшибка = {P_error_exp:.4f}',
             fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_exp[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
