import numpy as np
from scipy.stats import binom
from scipy import stats
import matplotlib.pyplot as plt

n = 5 * 7
M = 3
s = np.zeros((n, M))

# Буква Е
letterE = np.array([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 0,
    1, 0, 0, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

# Буква А
letterA = np.array([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1
])

# Буква Д
letterD = np.array([
    0, 1, 1, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1
])

s[:, 0] = letterE
s[:, 1] = letterA
s[:, 2] = letterD

pI = 0.65

def distort_image(image, pI):
    x = image.copy()
    r = np.random.rand(len(image))
    ir = r < pI
    x[ir] = 1 - x[ir]
    return x

# Создание первой визуализации
fig, axes = plt.subplots(3, 6, figsize=(18, 10))
letters = [letterE, letterA, letterD]
titles = ['Буква Е', 'Буква А', 'Буква Д']
num_distorted = 5

for i, (letter, title) in enumerate(zip(letters, titles)):
    axes[i, 0].imshow(letter.reshape(7, 5), cmap='binary', interpolation='nearest')
    axes[i, 0].set_title(f'{title}\n(Оригинал)', fontsize=12, weight='bold')
    axes[i, 0].axis('off')

    for j in range(num_distorted):
        distorted = distort_image(letter, pI)
        axes[i, j+1].imshow(distorted.reshape(7, 5), cmap='binary', interpolation='nearest')
        axes[i, j+1].set_title(f'Искажение {j+1}', fontsize=10)
        axes[i, j+1].axis('off')

plt.suptitle('Эталонные буквы и примеры искажений (pI = 0.65)',
             fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
plt.savefig('letters_templates.png', dpi=150, bbox_inches='tight')
plt.close()

pw = np.array([1/3, 1/3, 1/3])
K = 1000

# Штоб ошибок не было
if pI == 0:
    pI = 0.0001
if pI == 0.5:
    pI = 0.4999

pI_ = 1 - pI
s_ = 1 - s  # инвертированные изображения

Pc_ = np.zeros((M, M))  # экспериментальная
Pt = np.zeros((M, M))   # теоретическая

# Попарное сравнение классов
for ii in range(M - 1):
    for jj in range(ii + 1, M):
        # Вычисление порога принятия решений
        ns = int(np.sum(np.abs(s[:, ii] - s[:, jj])))
        l0_ = np.log(pw[jj] / pw[ii])
        L0 = np.log(pw[jj] / pw[ii]) / (2 * np.log(pI_) - 2 * np.log(pI)) + ns / 2
        L0r = int(np.floor(L0))

        # Определение вероятностей перепутывания
        if pI < 0.5:
            Pt[ii, jj] = binom.cdf(L0r, ns, 1 - pI)
            Pt[jj, ii] = 1 - binom.cdf(L0r, ns, pI)
        else:
            Pt[ii, jj] = 1 - binom.cdf(L0r, ns, 1 - pI)
            Pt[jj, ii] = binom.cdf(L0r, ns, pI)

for i in range(M):
    Pt[i, i] = 1 - np.sum(Pt[i, :])

for kk in range(K):
    for i in range(M):
        x = s[:, i].copy()
        r = np.random.rand(n)
        ir = r < pI
        x[ir] = 1 - x[ir]
        x_ = 1 - x

        iais = []

        for ii in range(M - 1):
            for jj in range(ii + 1, M):
                ns = int(np.sum(np.abs(s[:, ii] - s[:, jj])))
                l0_ = np.log(pw[jj] / pw[ii])
                L0 = np.log(pw[jj] / pw[ii]) / (2 * np.log(pI_) - 2 * np.log(pI)) + ns / 2
                L0r = int(np.floor(L0))

                # Вычисление коэффициентов разделяющей функции
                G1 = np.zeros(n)
                G2 = np.zeros(n)
                for k in range(n):
                    G1[k] = np.log((s[k, ii] * pI_ + s_[k, ii] * pI) /
                                   (s[k, jj] * pI_ + s_[k, jj] * pI))
                    G2[k] = np.log((s[k, ii] * pI + s_[k, ii] * pI_) /
                                   (s[k, jj] * pI + s_[k, jj] * pI_))

                # Классификация
                u = np.dot(G1, x) + np.dot(G2, x_) - l0_

                if u > 0:
                    iai = ii
                else:
                    iai = jj

                iais.append(iai)

        # Голосование большинством
        id = stats.mode(iais, keepdims=True)[0][0]

        Pc_[i, id] += 1

# Нормировка
Pc_ = Pc_ / K

diff = np.abs(Pt - Pc_)

P_error_theory = 1 - np.trace(Pt) / M
P_error_exp = 1 - np.trace(Pc_) / M

# Визуализация матриц
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Теоретическая матрица
im1 = axes[0].imshow(Pt, vmin=0, vmax=1)
axes[0].set_title('Теоретическая матрица ошибок', fontsize=14, weight='bold')
axes[0].set_xlabel('Распознан как класс', fontsize=12)
axes[0].set_ylabel('Истинный класс', fontsize=12)
axes[0].set_xticks([0, 1, 2])
axes[0].set_yticks([0, 1, 2])
axes[0].set_xticklabels(['Е (1)', 'А (2)', 'Д (3)'])
axes[0].set_yticklabels(['Е (1)', 'А (2)', 'Д (3)'])

for i in range(M):
    for j in range(M):
        text = axes[0].text(j, i, f'{Pt[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Экспериментальная матрица
im2 = axes[1].imshow(Pc_, vmin=0, vmax=1)
axes[1].set_title('Экспериментальная матрица ошибок', fontsize=14, weight='bold')
axes[1].set_xlabel('Распознан как класс', fontsize=12)
axes[1].set_ylabel('Истинный класс', fontsize=12)
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(['Е (1)', 'А (2)', 'Д (3)'])
axes[1].set_yticklabels(['Е (1)', 'А (2)', 'Д (3)'])

for i in range(M):
    for j in range(M):
        text = axes[1].text(j, i, f'{Pc_[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
