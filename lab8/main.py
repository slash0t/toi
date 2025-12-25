import numpy as np
import matplotlib.pyplot as plt

def make_bad_cov(n: int, eps: float) -> np.ndarray:
    """
    Ковариационная матрица для произвольного n.
    Последний признак сильно коррелирует с остальными => плохая обусловленность.
    """
    C = np.eye(n)
    C[-1, :] = (1 - eps) * np.ones(n)
    C[:, -1] = (1 - eps) * np.ones(n)
    C[-1, -1] = n - 1
    return C

def sample_correlated_normal(n: int, N: int, C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Генерация X ~ N(0, C).
    """
    jitter = 0.0
    for _ in range(6):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0 else jitter * 10

    Z = rng.standard_normal(size=(N, n))
    X = Z @ L.T
    return X

def ridge_solution(X: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """
    Ridge: (X^T X + beta I)^(-1) X^T y
    """
    XtX = X.T @ X
    p = XtX.shape[0]
    return np.linalg.solve(XtX + beta * np.eye(p), X.T @ y)

def residual_l2(X: np.ndarray, y: np.ndarray, a_hat: np.ndarray) -> float:
    """
    Невязка: sqrt( (Xa - y)^T (Xa - y) ) = ||Xa - y||_2
    """
    r = X @ a_hat - y
    return float(np.sqrt(r.T @ r))

def condition_number_sym(M: np.ndarray) -> float:
    """
    Число обусловленности симметричной матрицы по собственным значениям: lambda_max/lambda_min.
    """
    w = np.linalg.eigvalsh(M)
    w = np.sort(w)
    w = np.maximum(w, 0.0)
    # если совсем ноль, то условность бесконечна
    if w[0] <= 0:
        return np.inf
    return float(w[-1] / w[0])

def beta_for_target_cond(XtX: np.ndarray, target_cond: float) -> float:
    """
    Ищем beta >= 0 такое, что cond(XtX + beta I) = target_cond.
    """
    w = np.linalg.eigvalsh(XtX)
    w = np.sort(w)
    lam_min, lam_max = float(w[0]), float(w[-1])

    if lam_min <= 0:
        return float(max(1e-12, -lam_min + 1e-12))

    beta = (lam_max - target_cond * lam_min) / (target_cond - 1.0)
    return float(max(0.0, beta))


rng = np.random.default_rng(69)

n = 4 # размерность входа
eps = 1e-8 # степень обусловленности ковариации
D = 0.1 # дисперсия шума
xmin, xmax = -3, 3 # границы для новых данных
K = 10 # число шагов по beta
beta0 = 1e-7 # стартовая beta
target_cond = 500.0

Ns = [50, 200, 1000] # объёмы выборок по условию

A_true = np.array([-1, 1, 2, 3, 4], dtype=float)

C = make_bad_cov(n, eps)
cond_C = condition_number_sym(C)
print(f"cond(C) = {cond_C:.3e}")

betas_grid = beta0 * (10.0 ** np.arange(1, K + 1))
Mb = np.log10(betas_grid)

def delt_for_N(N: int) -> float:
    return 3.0 * np.sqrt(D * N)

results = {}

for N in Ns:
    XN = sample_correlated_normal(n, N, C, rng)
    X_train = np.hstack([np.ones((N, 1)), XN])
    y_train = X_train @ A_true + np.sqrt(D) * rng.standard_normal(N)

    XtX = X_train.T @ X_train
    cond_XtX = condition_number_sym(XtX)

    # Тестовые данные
    XN_test = rng.uniform(xmin, xmax, size=(N, n))
    X_test = np.hstack([np.ones((N, 1)), XN_test])
    y_test = X_test @ A_true + np.sqrt(D) * rng.standard_normal(N)

    # Находим beta*
    beta_star = beta_for_target_cond(XtX, target_cond)
    cond_reg = condition_number_sym(XtX + beta_star * np.eye(n + 1))

    # Анализ
    R_train = []
    R_test = []
    Ea = []

    for beta in betas_grid:
        a_hat = ridge_solution(X_train, y_train, beta)
        R_train.append(residual_l2(X_train, y_train, a_hat))
        R_test.append(residual_l2(X_test, y_test, a_hat))
        Ea.append(float(np.linalg.norm(A_true - a_hat)))

    R_train = np.array(R_train)
    R_test = np.array(R_test)
    Ea = np.array(Ea)

    # Метрики именно в точке beta*
    a_star = ridge_solution(X_train, y_train, beta_star)
    R_train_star = residual_l2(X_train, y_train, a_star)
    R_test_star = residual_l2(X_test, y_test, a_star)
    Ea_star = float(np.linalg.norm(A_true - a_star))

    results[N] = {
        "cond_XtX": cond_XtX,
        "beta_star": beta_star,
        "cond_reg": cond_reg,
        "grid": {
            "Mb": Mb,
            "betas": betas_grid,
            "R_train": R_train,
            "R_test": R_test,
            "Ea": Ea,
        },
        "at_star": {
            "R_train": R_train_star,
            "R_test": R_test_star,
            "Ea": Ea_star,
            "a_hat": a_star,
        }
    }


print("Итоги для beta*, где cond(XtX + beta*I) = 500")
for N in Ns:
    r = results[N]
    print(
        f"N={N:4d} | cond(XtX)={r['cond_XtX']:.3e} | "
        f"beta*={r['beta_star']:.3e} | cond_reg={r['cond_reg']:.3f} | "
        f"Rb_train={r['at_star']['R_train']:.3f} | "
        f"Rb_test={r['at_star']['R_test']:.3f} | "
        f"Ea={r['at_star']['Ea']:.3f}"
    )


fig = plt.figure(figsize=(14, 10))
fig.suptitle("Гребневая регрессия (n=4, eps=1e-8): Rb(train), Rb(test), Ea vs log10(beta)", fontsize=14)

for i, N in enumerate(Ns, start=1):
    ax = fig.add_subplot(3, 1, i)
    g = results[N]["grid"]
    beta_star = results[N]["beta_star"]
    delt = delt_for_N(N)

    ax.grid(True)
    ax.plot(g["Mb"], g["R_train"], marker="^", linestyle="-", label="Rb train")
    ax.plot(g["Mb"], g["R_test"], marker="^", linestyle="--", label="Rb test")
    ax.plot(g["Mb"], g["Ea"], marker="+", linestyle="--", label="Ea = ||A - a||")

    ax.plot(g["Mb"], np.full_like(g["Mb"], delt), linestyle="--", label="delt")

    ax.axvline(np.log10(beta_star) if beta_star > 0 else g["Mb"][0], linewidth=2)

    txt = (
        f"N={N},  cond(XtX)={results[N]['cond_XtX']:.2e}\n"
        f"beta*={beta_star:.2e},  cond_reg={results[N]['cond_reg']:.1f}\n"
        f"Rb_train*={results[N]['at_star']['R_train']:.2f},  "
        f"Rb_test*={results[N]['at_star']['R_test']:.2f},  "
        f"Ea*={results[N]['at_star']['Ea']:.2f}"
    )
    ax.text(g["Mb"][0] + 1, ax.get_ylim()[1] * 0.65, txt,
            bbox=dict(facecolor="white", alpha=0.85))

    ax.set_xlabel("log10(beta)")
    ax.set_ylabel("значение")
    ax.legend(loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 1])

# График метрик при beta* для всех N
fig2 = plt.figure(figsize=(12, 5))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.grid(True)
ax2.set_title("Метрики при beta* (cond_reg=500) для разных N")

Ns_arr = np.array(Ns)
Rb_tr = np.array([results[N]["at_star"]["R_train"] for N in Ns])
Rb_te = np.array([results[N]["at_star"]["R_test"] for N in Ns])
Ea_st = np.array([results[N]["at_star"]["Ea"] for N in Ns])

ax2.plot(Ns_arr, Rb_tr, marker="o", linestyle="-", label="Rb train @ beta*")
ax2.plot(Ns_arr, Rb_te, marker="o", linestyle="--", label="Rb test @ beta*")
ax2.plot(Ns_arr, Ea_st, marker="s", linestyle="-.", label="Ea @ beta*")

ax2.set_xlabel("N (объём выборки)")
ax2.set_ylabel("значение")
ax2.legend()

plt.show()
