"""
HARCH(2) Simulation Study — §1.3.4
Три блока:
  1. График параметрических областей на плоскости (a1, a2)
  2. Состоятельность MM и MLE при растущем N
  3. Сравнение эффективности: MSE vs граница Крамера–Рао
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize, fsolve
from matplotlib.colors import ListedColormap
from scipy.optimize import least_squares

rng = np.random.default_rng(42)

# ------------------------------------------------------------
# Вспомогательные функции HARCH(2)
# ------------------------------------------------------------
def sigma2(a0, a1, a2, h_prev1, h_prev2):
    return a0 + a1 * h_prev1**2 + a2 * (h_prev1 + h_prev2)**2

def simulate_harch2(a0, a1, a2, N, burn=200):
    h = np.zeros(N + burn + 2)
    eps = rng.standard_normal(N + burn + 2)
    for n in range(2, N + burn + 2):
        s2 = sigma2(a0, a1, a2, h[n-1], h[n-2])
        h[n] = np.sqrt(max(s2, 1e-12)) * eps[n]
    return h[burn+2:]

# ------------------------------------------------------------
# Условия существования моментов
# ------------------------------------------------------------
def m4_exists(a1, a2):
    if a2 >= 1: return False
    lhs = 3 * ((a1 + a2)**2 + a2**2) + a2 * (1 + 3*a1**2 + 18*a1*a2 + 12*a2**2)
    return lhs < 1

def m2_exists(a1, a2):
    return a1 + 2*a2 < 1

# ------------------------------------------------------------
# Метод моментов (MM)
# ------------------------------------------------------------
def mm_estimates(h):
    m2_hat = np.mean(h**2)
    R1_hat = np.mean(h[1:]**2 * h[:-1]**2)
    R2_hat = np.mean(h[2:]**2 * h[:-2]**2)

    def residuals(p):
        a0, a1, a2 = p
        # Штраф за выход за границы (чтобы не возвращать огромные значения)
        penalty = 0.0
        if a0 <= 0: penalty += 1e3 * (1 - a0)
        if a1 < 0: penalty += 1e3 * (-a1)
        if a2 < 0: penalty += 1e3 * (-a2)
        if a1 + 2*a2 >= 1: penalty += 1e3 * (a1 + 2*a2 - 0.999)
        if a2 >= 1: penalty += 1e3 * (a2 - 0.999)
        if penalty > 0:
            return [penalty, penalty, penalty]

        m2 = a0 / (1 - a1 - 2*a2)
        d = 1 - 3*((a1+a2)**2 + a2**2) - 6*a2*(a1+3*a2)*(a1+a2)/(1-a2)
        if d <= 0:
            return [1e3, 1e3, 1e3]
        num = 3*a0**2 + 6*a2*(a1+3*a2)*a0*m2/(1-a2) + 6*a0*(a1+2*a2)*m2
        m4 = num / d
        R1 = (a0*m2 + (a1+a2)*m4) / (1 - a2)
        R2 = a0*m2 + (a1+a2)*R1 + a2*m4
        # Нормированные невязки
        return [(m2 - m2_hat)/m2_hat, (R1 - R1_hat)/R1_hat, (R2 - R2_hat)/R2_hat]

    # Стартовые точки
    starts = [
        [m2_hat*0.95, 0.02, 0.01],
        [m2_hat*0.90, 0.05, 0.02],
        [m2_hat*0.75, 0.15, 0.05],
        [m2_hat*0.55, 0.25, 0.10],
        [m2_hat*0.40, 0.10, 0.15],
    ]
    best_res = None
    best_cost = np.inf
    for x0 in starts:
        # Обрезаем начальную точку в допустимые пределы (но не ниже 1e-6)
        x0[0] = max(x0[0], 1e-6)
        x0[1] = max(x0[1], 1e-6)
        x0[2] = max(x0[2], 1e-6)
        # Используем метод 'trf' с границами
        try:
            res = least_squares(residuals, x0, method='trf',
                                bounds=([1e-6, 0, 0], [np.inf, 1-1e-6, 0.5]),
                                max_nfev=300, ftol=1e-10, xtol=1e-10)
            if res.cost < best_cost and res.success:
                a0, a1, a2 = res.x
                if a0 > 1e-6 and a1 >= 0 and a2 >= 0 and a1+2*a2 < 1 and m4_exists(a1, a2):
                    best_cost = res.cost
                    best_res = res.x
        except Exception as e:
            continue
    return best_res if best_res is not None else np.array([np.nan]*3)

# ------------------------------------------------------------
# Метод максимального правдоподобия (MLE)
# ------------------------------------------------------------
def neg_loglik(params, h):
    a0, a1, a2 = params
    if a0 <= 1e-8 or a1 < 0 or a2 < 0 or a1 + 2*a2 >= 0.999:
        return 1e10
    N = len(h)
    ll = 0.0
    for n in range(2, N):
        s2 = sigma2(a0, a1, a2, h[n-1], h[n-2])
        if s2 <= 0:
            return 1e10
        ll += np.log(s2) + h[n]**2 / s2
    return 0.5 * ll

def mle_estimates(h):
    m2 = np.mean(h**2)
    # Нижняя граница a0 только положительная, верхней нет
    bounds = [(1e-6, None), (1e-6, 0.99), (1e-6, 0.49)]

    starts = []
    mm_start = mm_estimates(h)
    if not np.isnan(mm_start[0]) and mm_start[1]+2*mm_start[2] < 0.9:
        starts.append(list(mm_start))
    starts += [
        [m2*0.5, 0.05, 0.02],
        [m2*0.7, 0.15, 0.05],
        [m2*0.3, 0.10, 0.10],
    ]

    best = None
    best_nll = np.inf
    for x0 in starts:
        # Обрезаем только снизу
        x0 = [max(x0[0], 1e-6), max(x0[1], 1e-6), max(x0[2], 1e-6)]
        res = minimize(neg_loglik, x0, args=(h,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-10, 'maxiter': 500})
        if res.success and res.fun < best_nll:
            best_nll = res.fun
            best = res.x
    return best if best is not None else np.array([np.nan]*3)

# ------------------------------------------------------------
# Информационная матрица Фишера (численная)
# ------------------------------------------------------------
def fisher_info_numeric(a0, a1, a2, h):
    N = len(h)
    I = np.zeros((3,3))
    for n in range(2, N):
        s2 = sigma2(a0, a1, a2, h[n-1], h[n-2])
        xn = np.array([1.0, h[n-1]**2, (h[n-1]+h[n-2])**2])
        I += np.outer(xn, xn) / s2**2
    I /= (2 * N)
    return I

def plot_consistency():
    TRUE = (0.05, 0.10, 0.08)
    a0_true, a1_true, a2_true = TRUE
    N_vals = [500, 1000, 2000, 4000, 8000]
    n_rep = 300

    mm_means = {'a0':[], 'a1':[], 'a2':[]}
    mle_means = {'a0':[], 'a1':[], 'a2':[]}
    mm_std = {'a0':[], 'a1':[], 'a2':[]}
    mle_std = {'a0':[], 'a1':[], 'a2':[]}

    for N in N_vals:
        pairs = []   # список кортежей (mm_est, mle_est)
        valid = 0
        for _ in range(n_rep):
            h = simulate_harch2(a0_true, a1_true, a2_true, N)
            mm = mm_estimates(h)
            mle = mle_estimates(h)
            if not np.isnan(mm[0]) and not np.isnan(mle[0]):
                pairs.append((mm, mle))
                valid += 1
        if valid / n_rep < 0.8:
            print(f"Предупреждение: для N={N} только {valid}/{n_rep} реплик успешны.")
        pairs = np.array(pairs)
        if len(pairs) == 0:
            continue
        mm_arr = pairs[:,0,:]
        mle_arr = pairs[:,1,:]
        for idx, k in enumerate(['a0','a1','a2']):
            mm_vals = mm_arr[:, idx]
            mle_vals = mle_arr[:, idx]
            mm_means[k].append(np.mean(mm_vals))
            mle_means[k].append(np.mean(mle_vals))
            mm_std[k].append(np.std(mm_vals))
            mle_std[k].append(np.std(mle_vals))

    true_vals = {'a0':a0_true, 'a1':a1_true, 'a2':a2_true}
    labels = {'a0':r'$\hat{a}_0$', 'a1':r'$\hat{a}_1$', 'a2':r'$\hat{a}_2$'}
    fig, axes = plt.subplots(1,3, figsize=(13,4.5))

    for ax, k in zip(axes, ['a0','a1','a2']):
        ax.axhline(true_vals[k], color='black', lw=1.5, ls='--', label='Истинное значение')
        ax.plot(N_vals[:len(mm_means[k])], mm_means[k], 'o-', color='#e07b39', lw=2, ms=6, label='ММ (среднее)')
        ax.fill_between(N_vals[:len(mm_means[k])],
                         np.array(mm_means[k])-np.array(mm_std[k]),
                         np.array(mm_means[k])+np.array(mm_std[k]),
                         alpha=0.18, color='#e07b39')
        ax.plot(N_vals[:len(mle_means[k])], mle_means[k], 's-', color='#2b6cb0', lw=2, ms=6, label='ММП (среднее)')
        ax.fill_between(N_vals[:len(mle_means[k])],
                         np.array(mle_means[k])-np.array(mle_std[k]),
                         np.array(mle_means[k])+np.array(mle_std[k]),
                         alpha=0.18, color='#2b6cb0')
        ax.set_xscale('log')
        ax.set_xlabel('$N$', fontsize=12)
        ax.set_title(labels[k], fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Состоятельность ММ и ММП (общие реплики)  '
                 f'$(a_0={a0_true}, a_1={a1_true}, a_2={a2_true})$', fontsize=12)
    plt.tight_layout()
    plt.savefig('G:\\Study\\block2_consistency.png', dpi=150)
    plt.close()
    print("Блок 2 сохранён: block2_consistency.png")


def neg_loglik(params, h):
    a0, a1, a2 = params
    if a0 <= 1e-8 or a1 < 0 or a2 < 0 or a1 + 2 * a2 >= 0.999:
        return 1e10
    h_lag1 = h[1:-1]
    h_lag2 = h[:-2]
    h_now = h[2:]
    s2 = a0 + a1 * h_lag1 ** 2 + a2 * (h_lag1 + h_lag2) ** 2
    if np.any(s2 <= 0):
        return 1e10
    return 0.5 * np.sum(np.log(s2) + h_now ** 2 / s2)


from scipy.optimize import minimize

def mle_estimates(h):
    m2 = np.mean(h ** 2)
    a0_ub = max(2.0 * m2, 1e-3)                  # >>> ПРАВКА: a0 < m2, ставим 2*m2
    bounds = [(1e-6, a0_ub), (1e-6, 0.99), (1e-6, 0.49)]

    starts = []
    mm_start = mm_estimates(h)
    if not np.isnan(mm_start[0]) and mm_start[1] + 2 * mm_start[2] < 0.9:
        starts.append(list(mm_start))
    starts += [
        [m2 * 0.5, 0.05, 0.02],
        [m2 * 0.7, 0.15, 0.05],
        [m2 * 0.3, 0.10, 0.10],
    ]

    best, best_nll = None, np.inf
    for x0 in starts:
        x0 = [min(max(x0[0], 1e-6), a0_ub),
              min(max(x0[1], 1e-6), 0.99),
              min(max(x0[2], 1e-6), 0.49)]
        res = minimize(neg_loglik, x0, args=(h,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-10, 'maxiter': 500})
        if res.success and res.fun < best_nll:
            best_nll, best = res.fun, res.x
    return best if best is not None else np.array([np.nan] * 3)


def plot_efficiency():
    TRUE = (0.05, 0.10, 0.08)
    a0_true, a1_true, a2_true = TRUE
    N_vals = [500, 1000, 2000, 3000, 5000]
    n_rep = 400                                  # >>> ПРАВКА: было 100

    # --- CRB один раз по длинной траектории (без изменений) ---
    n_long, n_traj, I_sum = 500000, 20, 0
    for _ in range(n_traj):
        h_long = simulate_harch2(a0_true, a1_true, a2_true, n_long, burn=1000)
        I_sum += fisher_info_numeric(a0_true, a1_true, a2_true, h_long)
    Iinv = np.linalg.inv(I_sum / n_traj)

    mse_mm = {'a0': [], 'a1': [], 'a2': []}
    mse_mle = {'a0': [], 'a1': [], 'a2': []}
    cr_bound = {'a0': [], 'a1': [], 'a2': []}
    true_arr = np.array(TRUE)

    for N in N_vals:
        mm_list, mle_list = [], []
        for _ in range(n_rep):
            h = simulate_harch2(a0_true, a1_true, a2_true, N, burn=1000)
            mm_list.append(mm_estimates(h))
            mle_list.append(mle_estimates(h))
        mm_arr, mle_arr = np.array(mm_list), np.array(mle_list)
        # >>> ПРАВКА: НЕ парно — каждую оценку по СВОИМ валидным репликам
        for idx, k in enumerate(['a0', 'a1', 'a2']):
            mm_ok = mm_arr[~np.isnan(mm_arr[:, idx]), idx]
            mle_ok = mle_arr[~np.isnan(mle_arr[:, idx]), idx]
            mse_mm[k].append(np.mean((mm_ok - true_arr[idx]) ** 2))
            mse_mle[k].append(np.mean((mle_ok - true_arr[idx]) ** 2))
            cr_bound[k].append(Iinv[idx, idx] / N)

    # >>> ПРАВКА: численный контроль сходимости ММП к КР
    print("MSE(ММП)/CRB по N =", N_vals)
    for k in ['a0', 'a1', 'a2']:
        ratios = [m / c for m, c in zip(mse_mle[k], cr_bound[k])]
        print(f"  {k}: " + " ".join(f"{r:.2f}" for r in ratios))

    labels = {'a0': r'$a_0$', 'a1': r'$a_1$', 'a2': r'$a_2$'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, k in zip(axes, ['a0', 'a1', 'a2']):
        ax.plot(N_vals, mse_mm[k], 'o--', color='#e07b39', lw=2, ms=6, label='MSE (ММ)')
        ax.plot(N_vals, mse_mle[k], 's-', color='#2b6cb0', lw=2, ms=6, label='MSE (ММП)')
        ax.plot(N_vals, cr_bound[k], 'k-', lw=2, label='Граница КР')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$N$', fontsize=12)
        ax.set_title(f'MSE для {labels[k]}', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle(f'Эффективность ММ и ММП vs граница Крамера–Рао  '
                 f'$(a_0={a0_true}, a_1={a1_true}, a_2={a2_true})$, {n_rep} реплик',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('G:\\Study\\block3_efficiency.png', dpi=150)
    plt.close()
    print("Блок 3 сохранён: block3_efficiency.png")
# ------------------------------------------------------------
# Запуск
# ------------------------------------------------------------
if __name__ == '__main__':
    print("=== Блок 1: параметрические области ===")
    plot_parameter_regions()
    print("=== Блок 2: состоятельность ===")
    #plot_consistency()
    print("=== Блок 3: эффективность ===")
    plot_efficiency()
    print("\nГотово. Все графики сохранены.")