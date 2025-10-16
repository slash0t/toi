import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.linalg import inv, det
import warnings

warnings.filterwarnings('ignore')

# Настройка matplotlib для корректного отображения
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

n = 2  # размерность признакового пространства
M = 2  # число классов

# Математические ожидания классов
m1_base = np.array([2.0, 2.0])
m2_base = np.array([1.0, -1.0])

# Матрицы ковариации классов
C1 = np.array([[5.0, 1.0],
               [1.0, 5.0]])
C2 = np.array([[3.0, 1.0],
               [2.0, 4.0]])

# Априорные вероятности классов (равновероятные)
pw = np.array([0.5, 0.5])

# Вычисление обратных матриц и определителей
C1_inv = inv(C1)
C2_inv = inv(C2)
det_C1 = det(C1)
det_C2 = det(C2)

print(f"\nОпределитель det(C1) = {det_C1:.4f}")
print(f"Определитель det(C2) = {det_C2:.4f}")

# Расчет теоретических ошибок распознавания для двух классов.
def calculate_errors(m1, m2, C1, C2, pw):
    C1_inv = inv(C1)
    C2_inv = inv(C2)
    det_C1 = det(C1)
    det_C2 = det(C2)

    # Разность математических ожиданий
    dm = m1 - m2

    # Логарифм отношения априорных вероятностей
    l0 = np.log(pw[1] / pw[0])

    # Параметры для расчета ошибки P(2|1) - ошибка первого рода для класса 1
    tr12 = np.trace(C2_inv @ C1 - np.eye(n))
    tr12_2 = np.trace((C2_inv @ C1 - np.eye(n)) @ (C2_inv @ C1 - np.eye(n)))

    mg1 = 0.5 * (tr12 + dm.T @ C1_inv @ dm - np.log(det_C1 / det_C2))
    Dg1 = 0.5 * tr12_2 + dm.T @ C2_inv @ C1 @ C2_inv @ dm

    # Параметры для расчета ошибки P(1|2) - ошибка первого рода для класса 2
    tr21 = np.trace(np.eye(n) - C1_inv @ C2)
    tr21_2 = np.trace((np.eye(n) - C1_inv @ C2) @ (np.eye(n) - C1_inv @ C2))

    mg2 = 0.5 * (tr21 - dm.T @ C2_inv @ dm + np.log(det_C2 / det_C1))
    Dg2 = 0.5 * tr21_2 + dm.T @ C1_inv @ C2 @ C1_inv @ dm

    # Вероятности ошибок (теоретические)
    P12 = norm.cdf(l0, loc=mg1, scale=np.sqrt(Dg1))
    P21 = 1 - norm.cdf(l0, loc=mg2, scale=np.sqrt(Dg2))

    # Расстояние Бхаттачарьи
    C_avg = (C1 + C2) / 2
    mu2 = 0.125 * dm.T @ inv(C_avg) @ dm + 0.5 * np.log(det(C_avg) / np.sqrt(det_C1 * det_C2))

    # Границы Чернова
    P12_chernoff = np.sqrt(pw[1] / pw[0]) * np.exp(-mu2)
    P21_chernoff = np.sqrt(pw[0] / pw[1]) * np.exp(-mu2)

    return {
        'P12': P12,
        'P21': P21,
        'P12_chernoff': P12_chernoff,
        'P21_chernoff': P21_chernoff,
        'mu2': mu2,
        'mg1': mg1,
        'mg2': mg2,
        'Dg1': Dg1,
        'Dg2': Dg2
    }

# Метод статистических испытаний для оценки вероятностей ошибок.
def statistical_experiment(m1, m2, C1, C2, pw, K=5000):
    C1_inv = inv(C1)
    C2_inv = inv(C2)
    det_C1 = det(C1)
    det_C2 = det(C2)

    # Матрица вероятностей ошибок
    Pc = np.zeros((M, M))

    for k in range(K):
        for i in range(M):
            # Выбираем параметры для текущего класса
            if i == 0:
                mean = m1
                cov = C1
            else:
                mean = m2
                cov = C2

            # Генерация образа i-го класса
            x = np.random.multivariate_normal(mean, cov)

            # Вычисление значений разделяющих функций (дискриминантные функции)
            u = np.zeros(M)
            u[0] = -0.5 * (x - m1).T @ C1_inv @ (x - m1) - 0.5 * np.log(det_C1) + np.log(pw[0])
            u[1] = -0.5 * (x - m2).T @ C2_inv @ (x - m2) - 0.5 * np.log(det_C2) + np.log(pw[1])

            # Определение максимума (принятие решения о принадлежности к классу)
            j = np.argmax(u)
            Pc[i, j] += 1

    Pc = Pc / K
    return Pc


# Расчет базовых параметров
base_distance = np.linalg.norm(m1_base - m2_base)
print(f"Евклидово расстояние между классами: d = {base_distance:.4f}")

errors_base = calculate_errors(m1_base, m2_base, C1, C2, pw)

print(f"\nТеоретические вероятности ошибок:")
print(f"P(2|1) - ошибка первого рода для класса 1: {errors_base['P12']:.6f}")
print(f"P(1|2) - ошибка первого рода для класса 2: {errors_base['P21']:.6f}")

print(f"\nГраницы Чернова:")
print(f"P(2|1) Чернова для класса 1: {errors_base['P12_chernoff']:.6f}")
print(f"P(1|2) Чернова для класса 2: {errors_base['P21_chernoff']:.6f}")

print(f"\nРасстояние Бхаттачарьи: μ² = {errors_base['mu2']:.6f}")

# Статистический эксперимент для базовых параметров
print(f"\nПроведение статистического эксперимента (K = 10000 испытаний)...")
Pc_base = statistical_experiment(m1_base, m2_base, C1, C2, pw, K=10000)

print(f"\nЭкспериментальная матрица вероятностей ошибок:")
print(f"         Класс 1    Класс 2")
print(f"Класс 1  {Pc_base[0, 0]:.6f}  {Pc_base[0, 1]:.6f}")
print(f"Класс 2  {Pc_base[1, 0]:.6f}  {Pc_base[1, 1]:.6f}")

print(f"\nЭкспериментальная P(1|2) = {Pc_base[1, 0]:.6f}")
print(f"Теоретическая P(1|2) = {errors_base['P21']:.6f}")
print(f"Граница Чернова P(1|2) = {errors_base['P21_chernoff']:.6f}")

# Исследование зависимости ошибки от расстояния

# Создаем диапазон множителей для изменения расстояния
distance_factors = np.linspace(0.2, 3.5, 40)
distances = []
P21_theoretical = []
P21_chernoff = []


for factor in distance_factors:
    # Масштабируем вектор разности для изменения расстояния
    direction = (m2_base - m1_base) / np.linalg.norm(m2_base - m1_base)
    m2_new = m1_base + direction * base_distance * factor

    distance = np.linalg.norm(m1_base - m2_new)
    distances.append(distance)

    errors = calculate_errors(m1_base, m2_new, C1, C2, pw)
    P21_theoretical.append(errors['P21'])
    P21_chernoff.append(errors['P21_chernoff'])

print(f"\nДиапазон расстояний: от {min(distances):.4f} до {max(distances):.4f}")
print(f"Диапазон теоретических ошибок P(1|2): от {min(P21_theoretical):.6f} до {max(P21_theoretical):.6f}")
print(f"Диапазон границ Чернова P(1|2): от {min(P21_chernoff):.6f} до {max(P21_chernoff):.6f}")

# Статистические испытания нескольких точек

# Выберем 12 точек для эксперимента
experiment_indices = np.linspace(0, len(distances) - 1, 12, dtype=int)
P21_experimental = []
experiment_distances = []

for i, idx in enumerate(experiment_indices):
    factor = distance_factors[idx]
    direction = (m2_base - m1_base) / np.linalg.norm(m2_base - m1_base)
    m2_new = m1_base + direction * base_distance * factor

    distance = np.linalg.norm(m1_base - m2_new)
    experiment_distances.append(distance)

    # Эксперимент
    Pc = statistical_experiment(m1_base, m2_new, C1, C2, pw, K=5000)
    P21_experimental.append(Pc[1, 0])

    # Теоретическое значение для сравнения
    errors_curr = calculate_errors(m1_base, m2_new, C1, C2, pw)

    print(f"Точка {i + 1}: d={distance:.4f}, P(1|2)_эксп={Pc[1, 0]:.6f}, " +
          f"P(1|2)_теор={errors_curr['P21']:.6f}, " +
          f"P(1|2)_Чернова={errors_curr['P21_chernoff']:.6f}")

# Визуализация

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# График 1: Основной график - зависимость ошибки от расстояния
ax1 = axes[0, 0]
ax1.plot(distances, P21_theoretical, 'b-', linewidth=2, label='Теоретическое P(1|2)')
ax1.plot(distances, P21_chernoff, 'r--', linewidth=2, label='Оценка Чернова P(1|2)')
ax1.scatter(experiment_distances, P21_experimental, c='green', s=80,
            marker='o', label='Экспереминтальное P(1|2)', zorder=5, edgecolors='black')
ax1.axvline(x=base_distance, color='gray', linestyle=':', linewidth=1.5,
            label=f'Базовое расстояние = {base_distance:.2f}')
ax1.set_xlabel('Евклидово расстояние между классами', fontsize=12, fontweight='bold')
ax1.set_ylabel('Ошибка типа 1 вероятности P(1|2)', fontsize=12, fontweight='bold')
ax1.set_title('Тип 1 ошибки vs Расстояние между классами',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# График 2: Логарифмическая шкала для ошибок
ax2 = axes[0, 1]
ax2.semilogy(distances, P21_theoretical, 'b-', linewidth=2, label='Теоретическое P(1|2)')
ax2.semilogy(distances, P21_chernoff, 'r--', linewidth=2, label='Оценка Чернова P(1|2)')
ax2.scatter(experiment_distances, P21_experimental, c='green', s=80,
            marker='o', label='Экспериментальное P(1|2)', zorder=5, edgecolors='black')
ax2.set_xlabel('Евклидово расстояние между классами', fontsize=12, fontweight='bold')
ax2.set_ylabel('Тип 1 ошибки вероятности P(1|2) [log scale]', fontsize=12, fontweight='bold')
ax2.set_title('Ошибка типа 1 - логарифмическое масштабирование', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# График 3: Разность между границей Чернова и теоретическим значением
ax3 = axes[1, 0]
difference = np.array(P21_chernoff) - np.array(P21_theoretical)
ax3.plot(distances, difference, 'purple', linewidth=2, label='Чернова - Теоретическая')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.fill_between(distances, 0, difference, alpha=0.3, color='purple')
ax3.set_xlabel('Евклидово расстояние между классами', fontsize=12, fontweight='bold')
ax3.set_ylabel('Разница в ошибке вероятности', fontsize=12, fontweight='bold')
ax3.set_title('Разница: Оценки Чернова и теоретической',
              fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# График 4: Области классов и разделяющая граница для базовых параметров
ax4 = axes[1, 1]

# Создание сетки точек
x1_min = min(m1_base[0], m2_base[0]) - 6
x1_max = max(m1_base[0], m2_base[0]) + 6
x2_min = min(m1_base[1], m2_base[1]) - 6
x2_max = max(m1_base[1], m2_base[1]) + 6

x1_grid = np.linspace(x1_min, x1_max, 200)
x2_grid = np.linspace(x2_min, x2_max, 200)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
pos = np.dstack((X1, X2))

# Плотности распределения
rv1 = multivariate_normal(m1_base, C1)
rv2 = multivariate_normal(m2_base, C2)
pdf1 = rv1.pdf(pos)
pdf2 = rv2.pdf(pos)

# Отображение контуров плотностей
contour1 = ax4.contour(X1, X2, pdf1, levels=5, colors='blue', linewidths=1.5, alpha=0.7)
contour2 = ax4.contour(X1, X2, pdf2, levels=5, colors='red', linewidths=1.5, alpha=0.7)
ax4.clabel(contour1, inline=True, fontsize=8)
ax4.clabel(contour2, inline=True, fontsize=8)

# Разделяющая граница (решающая функция = 0)
decision_boundary = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        u1 = -0.5 * (x - m1_base).T @ C1_inv @ (x - m1_base) - 0.5 * np.log(det_C1) + np.log(pw[0])
        u2 = -0.5 * (x - m2_base).T @ C2_inv @ (x - m2_base) - 0.5 * np.log(det_C2) + np.log(pw[1])
        decision_boundary[i, j] = u1 - u2

ax4.contour(X1, X2, decision_boundary, levels=[0], colors='black', linewidths=2.5)

# Центры классов
ax4.plot(m1_base[0], m1_base[1], 'bs', markersize=12, label='Класс 1 среднее', markeredgecolor='black')
ax4.plot(m2_base[0], m2_base[1], 'r^', markersize=12, label='Класс 2 среднее', markeredgecolor='black')

ax4.set_xlabel('x1', fontsize=12, fontweight='bold')
ax4.set_ylabel('x2', fontsize=12, fontweight='bold')
ax4.set_title('Области классов и граница решений', fontsize=13, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

plt.tight_layout()
plt.show()
