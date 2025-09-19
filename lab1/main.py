import numpy as np
import matplotlib.pyplot as plt

a = 100
b = 200

theoretical_variance = b ** 2 / 12

def generate_uniform(a, b, size):
    alpha = np.random.rand(size)
    samples = a + b * alpha
    return samples

realizations = np.arange(100, 30001, 50)
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

plt.figure(figsize=(10, 6))
plt.plot(
    realizations,
    sample_variances,
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
plt.ylabel('Дисперсия')
plt.title(f'Зависимость дисперсии от числа реализаций для R({a}, {b})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('krutoy_graphic.png', dpi=150)
