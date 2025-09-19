import numpy as np
import matplotlib.pyplot as plt

# Параметры равномерного распределения
a = 100
b = 200

theoretical_variance = b ** 2 / 12 # Рассчет теоретической дисперсии

def generate_uniform(a, b, size):
    alpha = np.random.rand(size) # Генерация случайных величин
    samples = a + b * alpha # Преобразование к равномерному распределению
    return samples

realizations = np.arange(100, 100001, 100) # Все количества реализаций
sample_variances = [] # Выборочные дисперсии

print(f"Равномерное распределение R({a}, {b})")
print(f"Теоретическая дисперсия: {theoretical_variance:.4f}")
print("Генерация данных...")

for N in realizations:
    samples = generate_uniform(a, b, N) # Генерация выборки
    
    sample_var = np.var(samples) # Вычисление выборочной дисперсии
    sample_variances.append(sample_var) # Сохранение выборочной дисперсии
    
    if N % 5000 == 0:
        print(f"N = {N}: выборочная дисперсия = {sample_var:.4f}")

def plot_graph(x, y, filename, ylabel): # График зависимости дисперсии от числа реализаций
    plt.figure(figsize=(10, 6)) 
    plt.scatter(
        x,
        y,
        s=15,
        alpha=0.7,
        linewidth=1,
        label='Выборочная дисперсия',
    )
    plt.axhline(
        y=theoretical_variance,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Теоретическая дисперсия = {theoretical_variance:.4f}',
    )

    plt.xlabel('Количество реализаций')
    plt.ylabel(ylabel)
    plt.title(f'Зависимость дисперсии от числа реализаций для R({a}, {b})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=150)


plot_graph(realizations, sample_variances, 'krutoy_graphic.png', 'Дисперсия') # График зависимости дисперсии от числа реализаций
