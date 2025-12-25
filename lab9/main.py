import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from itertools import permutations

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
import warnings
warnings.filterwarnings("ignore")


def generate_data(n_features, n_classes, samples_per_class, means, dispersions, correlations, seed=42):
    """
    Генерация данных для классов на основе многомерного нормального распределения
    с ковариацией cov[i,j] = D * ro^|i-j|.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for k in range(n_classes):
        n = samples_per_class[k]
        mean = np.array(means[k], dtype=float)
        D = float(dispersions[k])
        ro = float(correlations[k])

        cov = np.zeros((n_features, n_features), dtype=float)
        for i in range(n_features):
            for j in range(n_features):
                cov[i, j] = D * (ro ** abs(i - j))

        samples = rng.multivariate_normal(mean=mean, cov=cov, size=n)
        X_list.append(samples)
        y_list.extend([k] * n)

    X = np.vstack(X_list)
    y_true = np.array(y_list, dtype=int)
    return X, y_true


def best_label_mapping(y_true, y_pred, n_classes):
    n = len(y_true)
    best_errors = n + 1
    best_perm = None

    for perm in permutations(range(n_classes)):
        mapped = np.zeros_like(y_pred)
        for cluster_id, class_id in enumerate(perm):
            mapped[y_pred == cluster_id] = class_id

        errors = np.sum(mapped != y_true)
        if errors < best_errors:
            best_errors = errors
            best_perm = perm

    return best_errors / n, best_perm


def apply_mapping(y_pred, mapping):
    mapped = np.zeros_like(y_pred)
    for cluster_id, class_id in enumerate(mapping):
        mapped[y_pred == cluster_id] = class_id
    return mapped


def run_hierarchical(X, y_true, n_classes, metric, linkage_method="average"):
    """
    Строим linkage, режем на n_classes кластеров, считаем error_rate и silhouette.
    """
    Z = linkage(X, method=linkage_method, metric=metric)
    y_pred = fcluster(Z, t=n_classes, criterion="maxclust") - 1

    err, mapping = best_label_mapping(y_true, y_pred, n_classes)
    y_mapped = apply_mapping(y_pred, mapping)

    sil = np.nan
    try:
        D = pairwise_distances(X, metric=metric)
        sil = silhouette_score(D, y_pred, metric="precomputed")
    except Exception:
        sil = np.nan

    return {
        "metric": metric,
        "Z": Z,
        "y_pred_raw": y_pred,
        "y_pred_mapped": y_mapped,
        "error_rate": err,
        "silhouette": sil,
        "mapping": mapping
    }


def plot_for_metric(X, y_true, res, linkage_method="average"):
    n_classes = len(np.unique(y_true))
    metric = res["metric"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_true, ax_pred, ax_den = axes

    markers = ["o", "^", "s", "D", "v", "<", ">", "p"]
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    # Истинные классы
    for k in range(n_classes):
        mask = (y_true == k)
        ax_true.scatter(X[mask, 0], X[mask, 1],
                        s=45, alpha=0.75,
                        marker=markers[k % len(markers)],
                        c=colors[k % len(colors)],
                        label=f"Класс {k+1}")
    ax_true.set_title("Истинные классы")
    ax_true.set_xlabel("x1")
    ax_true.set_ylabel("x2")
    ax_true.grid(True, alpha=0.25)
    ax_true.legend(fontsize=9)

    # Результат кластеризации
    y_mapped = res["y_pred_mapped"]
    for k in range(n_classes):
        mask = (y_mapped == k)
        ax_pred.scatter(X[mask, 0], X[mask, 1],
                        s=45, alpha=0.75,
                        marker=markers[k % len(markers)],
                        c=colors[k % len(colors)],
                        label=f"Класс {k+1}")
    ax_pred.set_title(
        f"Иерархическая ({metric})\n"
        f"err={res['error_rate']:.3f}, sil={res['silhouette']:.3f}"
    )
    ax_pred.set_xlabel("x1")
    ax_pred.set_ylabel("x2")
    ax_pred.grid(True, alpha=0.25)
    ax_pred.legend(fontsize=9)

    # Дендрограмма
    dendrogram(res["Z"], ax=ax_den,
               truncate_mode="lastp", p=25,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True)
    ax_den.set_title(f"Дендрограмма\nlinkage='{linkage_method}', metric='{metric}'")
    ax_den.set_xlabel("Группы/образцы (усечено)")
    ax_den.set_ylabel("Расстояние")
    ax_den.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(results_list):
    metrics = [r["metric"] for r in results_list]
    errors = [r["error_rate"] for r in results_list]
    sils = [(r["silhouette"] if not np.isnan(r["silhouette"]) else 0.0) for r in results_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    x = np.arange(len(metrics))

    ax1.bar(x, errors, edgecolor="black")
    ax1.set_title("Ошибка классификации по метрикам")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=25, ha="right")
    ax1.set_ylabel("Error rate")
    ax1.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(errors):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(x, sils, edgecolor="black")
    ax2.set_title("Silhouette score по метрикам")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=25, ha="right")
    ax2.set_ylabel("Silhouette")
    ax2.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(sils):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    n_features = 2
    n_classes = 4
    n_samples_per_class = 60
    dm = 4.0

    means = [
        [0, 0],
        [0, dm],
        [dm, 0],
        [dm, dm],
    ]
    dispersions = [1.0] * n_classes

    rng = np.random.default_rng(42)
    correlations = rng.uniform(-0.5, 0.5, n_classes).tolist()
    samples_per_class = [n_samples_per_class] * n_classes

    X, y_true = generate_data(
        n_features=n_features,
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        means=means,
        dispersions=dispersions,
        correlations=correlations,
        seed=42
    )

    # нормализация признаков
    X = StandardScaler().fit_transform(X)

    metrics = ["euclidean", "cityblock", "chebyshev", "cosine"]

    # Метод объединения
    linkage_method = "average"

    results = []
    for metric in metrics:
        res = run_hierarchical(X, y_true, n_classes, metric, linkage_method=linkage_method)
        results.append(res)

        print(f"[{metric}] error={res['error_rate']:.4f}, silhouette={res['silhouette']:.4f}, mapping={res['mapping']}")
        plot_for_metric(X, y_true, res, linkage_method=linkage_method)

    plot_metric_comparison(results)

    best = min(results, key=lambda r: r["error_rate"])
    print("\nЛучшая по ошибке метрика:")
    print(f"metric={best['metric']}, error={best['error_rate']:.4f}, silhouette={best['silhouette']:.4f}")


if __name__ == "__main__":
    main()
