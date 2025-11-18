import numpy as np
import matplotlib.pyplot as plt


# Математические ожидания классов
m1 = np.array([2.0, 2.0])
m2 = np.array([1.0, -1.0])

# Общая матрица ковариации
C = np.array([[5.0, 1.0],
              [1.0, 5.0]])


n_samples_per_class = 1000

RANDOM_SEED = 345621233

np.random.seed(RANDOM_SEED)

X1 = np.random.multivariate_normal(m1, C, n_samples_per_class)
X2 = np.random.multivariate_normal(m2, C, n_samples_per_class)

# Объединяем выборки
X = np.vstack((X1, X2))

# Метки классов: 0 для первого, 1 для второго
y = np.hstack((
    np.zeros(n_samples_per_class, dtype=int),
    np.ones(n_samples_per_class, dtype=int)
))


def confusion_matrix_prob(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          n_classes: int = 2) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1.0

    for i in range(n_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm[i] /= row_sum

    return cm


def knn_predict_one(x: np.ndarray,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    k: int) -> int:
    dists = np.linalg.norm(X_train - x, axis=1)
    idx = np.argpartition(dists, k)[:k]
    votes = y_train[idx]
    counts = np.bincount(votes, minlength=2)
    return int(np.argmax(counts))


def knn_predict(X_test: np.ndarray,
                X_train: np.ndarray,
                y_train: np.ndarray,
                k: int) -> np.ndarray:
    return np.array([knn_predict_one(x, X_train, y_train, k) for x in X_test])


def sliding_cv_confusion_knn(X: np.ndarray,
                             y: np.ndarray,
                             a: float,
                             power: float,
                             n_folds: int = 10,
                             random_state: int = 123):
    n = len(X)
    indices = np.arange(n)

    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    current = 0

    cms = []
    used_ks = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        current = stop

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        N_train = len(X_train)

        # k = a * N^power
        k_raw = int(round(a * (N_train ** power)))

        # Коррекция k: 1 <= k <= N_train, предпочтительно нечётное
        if k_raw < 1:
            k_raw = 1
        if k_raw % 2 == 0:
            k_raw += 1
        if k_raw > N_train:
            k_raw = N_train if (N_train % 2 == 1) else (N_train - 1)

        used_ks.append(k_raw)

        y_pred = knn_predict(X_test, X_train, y_train, k_raw)
        cm = confusion_matrix_prob(y_test, y_pred, n_classes=2)
        cms.append(cm)

    mean_cm = np.mean(cms, axis=0)
    return mean_cm, np.array(used_ks)


param_settings = [
    {"a": 1, "power": 0.5},
    {"a": 2, "power": 0.1},
    {"a": 2, "power": 0.5},
    {"a": 0.50, "power": 0.1},
]

results = {}  # (a, power) -> dict с матрицей ошибок и статистикой по k

for params in param_settings:
    a = params["a"]
    power = params["power"]

    mean_cm, used_ks = sliding_cv_confusion_knn(
        X, y, a, power, n_folds=10, random_state=123
    )

    mean_error = 0.5 * (mean_cm[0, 1] + mean_cm[1, 0])

    results[(a, power)] = {
        "confusion": mean_cm,
        "error": mean_error,
        "k_mean": used_ks.mean(),
        "k_min": used_ks.min(),
        "k_max": used_ks.max(),
        "k_all": used_ks,
    }

best_params, best_info = min(
    results.items(),
    key=lambda item: item[1]["error"]
)

best_a, best_power = best_params
best_cm = best_info["confusion"]
best_error = best_info["error"]

n_plots = len(results)
fig, axes = plt.subplots(
    1, n_plots, figsize=(5 * n_plots, 4)
)

if n_plots == 1:
    axes = [axes]

last_im = None

for ax, ((a, power), info) in zip(axes, results.items()):
    cm = info["confusion"]
    k_mean = int(round(info["k_mean"]))  # итоговое k (среднее по фолдам)

    im = ax.imshow(cm, vmin=0, vmax=1)
    last_im = im  # запоминаем для colorbar

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Класс 0", "Класс 1"])
    ax.set_yticklabels(["Класс 0", "Класс 1"])
    ax.set_xlabel("Решение")
    ax.set_ylabel("Истинный класс")

    # Заголовок: a, power, средняя ошибка и итоговое k
    ax.set_title(
        f"a={a:.2f}, y={power:.2f}, k≈{k_mean}\n"
        f"err={info['error']:.3f}"
    )

    # Подписи внутри ячеек
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:.3f}",
                ha="center", va="center", color="black"
            )

plt.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(last_im, cax=cbar_ax)

plt.show()
