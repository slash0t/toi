import numpy as np
import matplotlib.pyplot as plt

a = 100
b = 200

theoretical_variance = b ** 2 / 12

def generate_uniform(a, b, size):
    alpha = np.random.rand(size)
    samples = a + b * alpha
    return samples

realizations = np.arange(100, 1000001, 100)
sample_variances = []

print(f"Равномерное распределение R({a}, {b})")
print(f"Теоретическая дисперсия: {theoretical_variance:.4f}")
print("Генерация данных...")

for N in realizations:
    samples = generate_uniform(a, b, N)
    
    sample_var = np.var(samples, ddof=1)
    sample_variances.append(sample_var)
    
    if N % 5000 == 0:
        print(f"N = {N}: выборочная дисперсия = {sample_var:.4f}")

def plot_graph(x, y, filename, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        y,
        'b-',
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


plot_graph(realizations, sample_variances, 'krutoy_graphic.png', 'Дисперсия')

# window = 5
# averaged_realizations = []
# averaged_sample_variances = []

# for i in range(0, len(realizations), 5):
#     start = i
#     end = min(start + 5, len(realizations))
#     cnt = end - start
#     averaged_realizations.append(sum(realizations[start:end]) / cnt)
#     averaged_sample_variances.append(sum(sample_variances[start:end]) / cnt)
# print(averaged_realizations[-1])
# print(averaged_sample_variances[-1])

# plot_graph(averaged_realizations, averaged_sample_variances, 'krutoy_graphic2.png', f'Средняя дисперсия по окнам {window}')
