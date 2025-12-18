# lab7_svm_kernels.py
# ЛР №7: SVM. Оценка вероятностей ошибок для линейно НЕразделимых выборок (2 класса)
# Ядра: квадратичная функция (poly degree=2) и mlp (sigmoid). Выбор оптимального ядра.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score


# ---------------------------
# 1) Генерация данных (две гауссовские выборки с перекрытием => НЕлинейно разделимы)
# ---------------------------
def generate_gaussian_2class(n_samples, priors, means, covs, rng):
    """
    n_samples: общий размер выборки
    priors: [p0, p1]
    means: [mu0 (2,), mu1 (2,)]
    covs:  [C0 (2,2), C1 (2,2)]
    """
    priors = np.array(priors, dtype=float)
    priors = priors / priors.sum()

    n0 = int(np.floor(n_samples * priors[0]))
    n1 = n_samples - n0

    X0 = rng.multivariate_normal(mean=means[0], cov=covs[0], size=n0)
    X1 = rng.multivariate_normal(mean=means[1], cov=covs[1], size=n1)

    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------
# 2) Обучение + оценка вероятностей ошибок
# ---------------------------
def fit_best_svm(X_train, y_train, kernel_name, param_grid, seed):
    """
    Подбор гиперпараметров через CV, чтобы сравнение ядер было честным.
    """
    base = SVC(kernel=kernel_name)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def error_probabilities(y_true, y_pred):
    """
    Возвращает:
    - матрицу P(решение=j | истинный=i) (нормированная confusion matrix по строкам)
    - P_err|0 = P(ŷ!=0 | y=0)
    - P_err|1 = P(ŷ!=1 | y=1)
    - P_err_overall = 1 - accuracy
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # по строкам: условные вероятности решений

    p_err_0 = 1.0 - cm_norm[0, 0]
    p_err_1 = 1.0 - cm_norm[1, 1]
    p_err_overall = 1.0 - accuracy_score(y_true, y_pred)

    return cm_norm, p_err_0, p_err_1, p_err_overall


# ---------------------------
# 3) Визуализация областей решений (как в MATLAB примере)
# ---------------------------
def plot_decision_regions_2d(ax, clf, X, y, title, step=0.05):
    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, step),
                         np.arange(x2_min, x2_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.25)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, label="class 1")

    if hasattr(clf, "support_vectors_"):
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=70, facecolors="none", edgecolors="k", label="SV")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")


# ---------------------------
# 4) Основной эксперимент (статистические испытания)
# ---------------------------
def main():
    # --- Настройки эксперимента ---
    SEED = 7
    rng = np.random.default_rng(SEED)

    # Две гауссовские выборки (перекрываются) => линейно НЕразделимы
    priors = [0.4, 0.6]

    means = [
        np.array([-2.0, -2.0]),
        np.array([ 2.0,  2.0]),
    ]

    covs = [
        np.array([[3.0, -1.0],
                  [-1.0, 3.0]]),
        np.array([[5.0,  3.0],
                  [3.0,  5.0]]),
    ]

    N_TRIALS = 3      # число статистических испытаний
    N_TRAIN  = 20     # объем обучающей выборки
    N_TEST   = 400    # объем тестовой выборки (больше => точнее вероятности)

    # --- Ядра и сетки гиперпараметров ---
    # Квадратичная функция ядра: polynomial degree=2
    grid_quadratic = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.01, 0.1, 1, 10],
        "coef0": [0, 1, 5],
        "degree": [2],
    }

    # "mlp" ядро (в классическом SVM это sigmoid, как в LIBSVM)
    grid_mlp = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.01, 0.1, 1, 10],
        "coef0": [-1, 0, 1, 5],
    }

    kernels = [
        ("quadratic (poly d=2)", "poly", grid_quadratic),
        ("mlp (sigmoid)",        "sigmoid", grid_mlp),
    ]

    rows = []

    # Чтобы построить картинки — сохраним первый прогон
    first_trial_models = {}

    for t in range(N_TRIALS):
        X_train, y_train = generate_gaussian_2class(N_TRAIN, priors, means, covs, rng)
        X_test,  y_test  = generate_gaussian_2class(N_TEST,  priors, means, covs, rng)

        for title, kernel_name, grid in kernels:
            clf, best_params, cv_acc = fit_best_svm(
                X_train, y_train, kernel_name, grid, seed=SEED + t
            )
            y_pred = clf.predict(X_test)

            cm_norm, p_err_0, p_err_1, p_err_overall = error_probabilities(y_test, y_pred)

            rows.append({
                "trial": t,
                "kernel": title,
                "cv_acc": cv_acc,
                "acc_test": 1 - p_err_overall,
                "P(err|0)": p_err_0,
                "P(err|1)": p_err_1,
                "P(err)": p_err_overall,
                "best_params": best_params,
            })

            if t == 0:
                first_trial_models[title] = (clf, X_train, y_train)

    df = pd.DataFrame(rows)

    # --- Итоговые оценки вероятностей ошибок (среднее по испытаниям) ---
    summary = df.groupby("kernel")[["P(err|0)", "P(err|1)", "P(err)", "acc_test", "cv_acc"]].mean()
    stds = df.groupby("kernel")[["P(err|0)", "P(err|1)", "P(err)"]].std()

    print("\n=== Средние вероятности ошибок (по испытаниям) ===")
    print(summary)
    print("\n=== Стандартные отклонения вероятностей ошибок ===")
    print(stds)

    # --- Выбор оптимального ядра ---
    best_kernel = summary["P(err)"].idxmin()
    print(f"\nОптимальное ядро по критерию min P(err): {best_kernel}")

    # --- Графики ---
    # 1) Decision regions для первого прогона (2 подграфика)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, _, _) in zip(axes, kernels):
        clf, Xtr, ytr = first_trial_models[title]
        plot_decision_regions_2d(ax, clf, Xtr, ytr, title=f"Decision regions: {title}")
    plt.tight_layout()
    plt.show()

    # 2) Столбчатая диаграмма средних P(err)
    fig = plt.figure(figsize=(7, 4))
    plt.bar(summary.index, summary["P(err)"])
    plt.ylabel("P(err) = 1 - Accuracy")
    plt.title("Сравнение средних вероятностей ошибок")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
