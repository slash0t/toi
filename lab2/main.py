import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

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

print(f"Расстояние Махаланобиса между классами: {d_mahal:.4f}")
print()

# Вычисляем параметры решающей функции
w = C_inv @ (m1 - m2)  # вектор весов
w0 = -0.5 * (m1.T @ C_inv @ m1 - m2.T @ C_inv @ m2) + np.log(P2/P1)

print("Решающее правило (линейный дискриминант):")
print(f"Веса w: {w}")
print(f"Смещение w0: {w0:.4f}")
print()

# Проекция на направление различения классов
# Стандартное отклонение проекции
sigma_proj = np.sqrt((m1 - m2).T @ C @ (m1 - m2) / (d_mahal**2))

# Теоретическая вероятность ошибки
P_error_theory = norm.cdf(-d_mahal / 2)

print(f"Теоретическая вероятность ошибки: {P_error_theory:.6f}")
print(f"Теоретическая точность распознавания: {1 - P_error_theory:.6f}")
print()

# Матрица ошибок (confusion matrix) - теоретическая
# P(error | Class 1) = P(classify as Class 2 | true Class 1)
P_error_given_1 = norm.cdf(-d_mahal / 2)
P_error_given_2 = norm.cdf(-d_mahal / 2)

confusion_theory = np.array([
    [1 - P_error_given_1, P_error_given_1],  # True Class 1
    [P_error_given_2, 1 - P_error_given_2]   # True Class 2
])

print("Теоретическая матрица ошибок (confusion matrix):")
print("                  Predicted Class 1  Predicted Class 2")
print(f"True Class 1:         {confusion_theory[0,0]:.6f}         {confusion_theory[0,1]:.6f}")
print(f"True Class 2:         {confusion_theory[1,0]:.6f}         {confusion_theory[1,1]:.6f}")
print()
print(f"Ошибка 1-го рода (False Positive): {confusion_theory[1,0]:.6f}")
print(f"Ошибка 2-го рода (False Negative): {confusion_theory[0,1]:.6f}")
print()

# Вычислительный эксперимент

# Генерация обучающей выборки
np.random.seed(52)
n_samples = 1000

# Генерация выборок из каждого класса
X1 = np.random.multivariate_normal(m1, C, n_samples)
X2 = np.random.multivariate_normal(m2, C, n_samples)

print(f"Сгенерировано {n_samples} образцов для каждого класса")
print()

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

print("Экспериментальная матрица ошибок (confusion matrix):")
print("                  Predicted Class 1  Predicted Class 2")
print(f"True Class 1:         {confusion_exp[0,0]:.6f}         {confusion_exp[0,1]:.6f}")
print(f"True Class 2:         {confusion_exp[1,0]:.6f}         {confusion_exp[1,1]:.6f}")
print()

# Общая точность
accuracy_exp = (correct_1 + correct_2) / 2
error_exp = (error_1 + error_2) / 2

print(f"Экспериментальная точность: {accuracy_exp:.6f}")
print(f"Экспериментальная ошибка: {error_exp:.6f}")
print()

print("Сравнение теории и эксперимента:")
print(f"Теоретическая ошибка: {P_error_theory:.6f}")
print(f"Экспериментальная ошибка: {error_exp:.6f}")
print(f"Относительная разница: {abs(P_error_theory - error_exp) / P_error_theory * 100:.2f}%")
print()

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
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# График 1: Распределение классов и решающая граница
ax = axes[0, 0]
contour = ax.contourf(xx, yy, Z, alpha=0.3, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'])
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', s=20, alpha=0.5, label='Класс 1 (истинный)')
ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='s', s=20, alpha=0.5, label='Класс 2 (истинный)')
ax.scatter(m1[0], m1[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2,
           label='Центр класса 1', zorder=5)
ax.scatter(m2[0], m2[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2,
           label='Центр класса 2', zorder=5)
ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title('Распределение классов и решающая граница', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# График 2: Изолинии плотностей вероятности
ax = axes[0, 1]

# Вычисление плотностей
x_grid = np.linspace(x_min, x_max, 200)
y_grid = np.linspace(y_min, y_max, 200)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_grid, Y_grid))

rv1 = multivariate_normal(m1, C)
rv2 = multivariate_normal(m2, C)

pdf1 = rv1.pdf(pos)
pdf2 = rv2.pdf(pos)

contour1 = ax.contour(X_grid, Y_grid, pdf1, levels=5, colors='red', alpha=0.6)
contour2 = ax.contour(X_grid, Y_grid, pdf2, levels=5, colors='blue', alpha=0.6)
ax.clabel(contour1, inline=True, fontsize=8)
ax.clabel(contour2, inline=True, fontsize=8)

ax.scatter(m1[0], m1[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2,
           label='Центр класса 1', zorder=5)
ax.scatter(m2[0], m2[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2,
           label='Центр класса 2', zorder=5)

ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title('Изолинии плотностей вероятности', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# График 3: Матрица ошибок (теоретическая)
ax = axes[1, 0]
im = ax.imshow(confusion_theory, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_yticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title('Теоретическая матрица ошибок', fontsize=14, fontweight='bold')

# Добавление значений в ячейки
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_theory[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# График 4: Матрица ошибок (экспериментальная)
ax = axes[1, 1]
im = ax.imshow(confusion_exp, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_yticklabels(['Класс 1', 'Класс 2'], fontsize=11)
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title('Экспериментальная матрица ошибок', fontsize=14, fontweight='bold')

# Добавление значений в ячейки
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_exp[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Подзадача b

# Новые параметры - центры ближе друг к другу
m1_b = np.array([1.5, 0.5])
m2_b = np.array([1, -0.5])
C_b = C  # Матрицу ковариации оставляем той же

# Теоретический анализ для нового случая
C_b_inv = np.linalg.inv(C_b)
delta_m_b = m1_b - m2_b
d_mahal_b = np.sqrt(delta_m_b.T @ C_b_inv @ delta_m_b)

print(f"Новое расстояние Махаланобиса: {d_mahal_b:.4f}")
print(f"Старое расстояние Махаланобиса: {d_mahal:.4f}")
print(f"Изменение: {((d_mahal_b - d_mahal) / d_mahal * 100):.2f}%")
print()

P_error_theory_b = norm.cdf(-d_mahal_b / 2)
print(f"Новая теоретическая ошибка: {P_error_theory_b:.6f}")
print(f"Старая теоретическая ошибка: {P_error_theory:.6f}")
print(f"Увеличение ошибки: {((P_error_theory_b - P_error_theory) / P_error_theory * 100):.2f}%")
print()

# Вычислительный эксперимент
X1_b = np.random.multivariate_normal(m1_b, C_b, n_samples)
X2_b = np.random.multivariate_normal(m2_b, C_b, n_samples)

y1_pred_b = np.array([classify(x, m1_b, m2_b, C_b_inv, P1, P2) for x in X1_b])
y2_pred_b = np.array([classify(x, m1_b, m2_b, C_b_inv, P1, P2) for x in X2_b])

error_exp_b = ((np.sum(y1_pred_b == 2) + np.sum(y2_pred_b == 1)) / (2 * n_samples))

print(f"Новая экспериментальная ошибка: {error_exp_b:.6f}")
print(f"Старая экспериментальная ошибка: {error_exp:.6f}")
print(f"Увеличение ошибки: {((error_exp_b - error_exp) / error_exp * 100):.2f}%")
print()

# Визуализация для задания (b)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Старые данные
ax = axes[0]
x_min_b = min(X1[:, 0].min(), X2[:, 0].min(), X1_b[:, 0].min(), X2_b[:, 0].min()) - 2
x_max_b = max(X1[:, 0].max(), X2[:, 0].max(), X1_b[:, 0].max(), X2_b[:, 0].max()) + 2
y_min_b = min(X1[:, 1].min(), X2[:, 1].min(), X1_b[:, 1].min(), X2_b[:, 1].min()) - 2
y_max_b = max(X1[:, 1].max(), X2[:, 1].max(), X1_b[:, 1].max(), X2_b[:, 1].max()) + 2

xx_b, yy_b = np.meshgrid(np.linspace(x_min_b, x_max_b, 200),
                         np.linspace(y_min_b, y_max_b, 200))
Z_old = np.array([classify(np.array([x, y]), m1, m2, C_inv, P1, P2)
                  for x, y in zip(xx_b.ravel(), yy_b.ravel())])
Z_old = Z_old.reshape(xx_b.shape)

ax.contourf(xx_b, yy_b, Z_old, alpha=0.3, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'])
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', s=20, alpha=0.5, label='Класс 1')
ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='s', s=20, alpha=0.5, label='Класс 2')
ax.scatter(m1[0], m1[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.scatter(m2[0], m2[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title(f'Исходные данные\nОшибка: {error_exp:.4f}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Новые данные
ax = axes[1]
Z_new = np.array([classify(np.array([x, y]), m1_b, m2_b, C_b_inv, P1, P2)
                  for x, y in zip(xx_b.ravel(), yy_b.ravel())])
Z_new = Z_new.reshape(xx_b.shape)

ax.contourf(xx_b, yy_b, Z_new, alpha=0.3, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'])
ax.scatter(X1_b[:, 0], X1_b[:, 1], c='red', marker='o', s=20, alpha=0.5, label='Класс 1')
ax.scatter(X2_b[:, 0], X2_b[:, 1], c='blue', marker='s', s=20, alpha=0.5, label='Класс 2')
ax.scatter(m1_b[0], m1_b[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.scatter(m2_b[0], m2_b[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title(f'Изменённые данные (задание b)\nОшибка: {error_exp_b:.4f}',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Подзадача c
# Увеличить ошибку 1 рода, уменьшить второй род
m1_c = m1
m2_c = m2
C_c = C

P1_c = 0.8
P2_c = 0.2

# Теоретический анализ
C_c_inv = np.linalg.inv(C_c)
delta_m_c = m1_c - m2_c
d_mahal_c = np.sqrt(delta_m_c.T @ C_c_inv @ delta_m_c)

print(f"Расстояние Махаланобиса: {d_mahal_c:.4f}")
print()

# Вычислительный эксперимент
np.random.seed(52)
X1_c = np.random.multivariate_normal(m1_c, C_c, n_samples)
X2_c = np.random.multivariate_normal(m2_c, C_c, n_samples)

y1_pred_c = np.array([classify(x, m1_c, m2_c, C_c_inv, P1_c, P2_c) for x in X1_c])
y2_pred_c = np.array([classify(x, m1_c, m2_c, C_c_inv, P1_c, P2_c) for x in X2_c])

# Матрица ошибок
error_1_roda_c = np.sum(y2_pred_c == 1) / n_samples  # FP
error_2_roda_c = np.sum(y1_pred_c == 2) / n_samples  # FN

confusion_exp_c = np.array([
    [np.sum(y1_pred_c == 1) / n_samples, error_2_roda_c],
    [error_1_roda_c, np.sum(y2_pred_c == 2) / n_samples]
])

print("Матрица ошибок для задания (c):")
print("                  Predicted Class 1  Predicted Class 2")
print(f"True Class 1:         {confusion_exp_c[0,0]:.6f}         {confusion_exp_c[0,1]:.6f}")
print(f"True Class 2:         {confusion_exp_c[1,0]:.6f}         {confusion_exp_c[1,1]:.6f}")
print()

print("Сравнение с исходными данными:")
print(f"Ошибка 1-го рода (FP):")
print(f"  Исходная: {confusion_exp[1,0]:.6f}")
print(f"  Новая:    {error_1_roda_c:.6f}")
print(f"  Изменение: {((error_1_roda_c - confusion_exp[1,0]) / max(confusion_exp[1,0], 1e-10) * 100):+.2f}%")
print()
print(f"Ошибка 2-го рода (FN):")
print(f"  Исходная: {confusion_exp[0,1]:.6f}")
print(f"  Новая:    {error_2_roda_c:.6f}")
print(f"  Изменение: {((error_2_roda_c - confusion_exp[0,1]) / max(confusion_exp[0,1], 1e-10) * 100):+.2f}%")
print()

# Визуализация для задания (c)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Исходные данные с матрицей ошибок
ax = axes[0]
x_min_c = min(X1[:, 0].min(), X2[:, 0].min(), X1_c[:, 0].min(), X2_c[:, 0].min()) - 2
x_max_c = max(X1[:, 0].max(), X2[:, 0].max(), X1_c[:, 0].max(), X2_c[:, 0].max()) + 2
y_min_c = min(X1[:, 1].min(), X2[:, 1].min(), X1_c[:, 1].min(), X2_c[:, 1].min()) - 2
y_max_c = max(X1[:, 1].max(), X2[:, 1].max(), X1_c[:, 1].max(), X2_c[:, 1].max()) + 2

xx_c, yy_c = np.meshgrid(np.linspace(x_min_c, x_max_c, 200),
                         np.linspace(y_min_c, y_max_c, 200))
Z_old_c = np.array([classify(np.array([x, y]), m1, m2, C_inv, P1, P2)
                    for x, y in zip(xx_c.ravel(), yy_c.ravel())])
Z_old_c = Z_old_c.reshape(xx_c.shape)

ax.contourf(xx_c, yy_c, Z_old_c, alpha=0.3, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'])
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', s=20, alpha=0.5, label='Класс 1')
ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='s', s=20, alpha=0.5, label='Класс 2')
ax.scatter(m1[0], m1[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.scatter(m2[0], m2[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title(f'Исходные данные\nОшибка 1 рода: {confusion_exp[1,0]:.4f}\nОшибка 2 рода: {confusion_exp[0,1]:.4f}',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Новые данные для задания (c)
ax = axes[1]
Z_new_c = np.array([classify(np.array([x, y]), m1_c, m2_c, C_c_inv, P1_c, P2_c)
                    for x, y in zip(xx_c.ravel(), yy_c.ravel())])
Z_new_c = Z_new_c.reshape(xx_c.shape)

ax.contourf(xx_c, yy_c, Z_new_c, alpha=0.3, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'])
ax.scatter(X1_c[:, 0], X1_c[:, 1], c='red', marker='o', s=20, alpha=0.5, label='Класс 1')
ax.scatter(X2_c[:, 0], X2_c[:, 1], c='blue', marker='s', s=20, alpha=0.5, label='Класс 2')
ax.scatter(m1_c[0], m1_c[1], c='darkred', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.scatter(m2_c[0], m2_c[1], c='darkblue', marker='*', s=500, edgecolors='black', linewidths=2, zorder=5)
ax.set_xlabel('Признак 1', fontsize=12)
ax.set_ylabel('Признак 2', fontsize=12)
ax.set_title(f'Изменённые данные (задание c)\nОшибка 1 рода: {error_1_roda_c:.4f}\nОшибка 2 рода: {error_2_roda_c:.4f}',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
