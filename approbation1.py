import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

# ---------- данные: USD/RUB, фиксинг ЦБ, колонка CLOSE ----------
df = pd.read_csv('USDCB_250128_260228.csv', sep=';')
df.columns = [c.strip('<>').upper() for c in df.columns]
close = df['CLOSE'].astype(float).values
h = np.diff(np.log(close))
h = h - h.mean()                      # центрирование (mu_n = 0)
N = len(h)

# ---------- условная дисперсия HARCH(2) ----------
def sigma2(theta, h):
    a0, a1, a2 = theta
    s = np.empty(len(h))
    for t in range(len(h)):
        h1 = h[t-1] if t >= 1 else 0.0
        h2 = h[t-2] if t >= 2 else 0.0
        s[t] = a0 + a1*h1**2 + a2*(h1 + h2)**2
    return s

# ---------- ММП (гауссова спецификация) ----------
def neg_loglik(theta, h, start=2):
    a0, a1, a2 = theta
    if a0 <= 0 or a1 < 0 or a2 < 0 or a1 + 2*a2 >= 1:
        return 1e12
    s = np.maximum(sigma2(theta, h)[start:], 1e-300)
    hs = h[start:]
    return 0.5*np.sum(np.log(2*np.pi) + np.log(s) + hs**2/s)

m2hat = np.mean(h**2)
best = None
for a0s in (0.1, 0.5, 1.0):
    for a1s in (0.0, 0.05, 0.15, 0.3):
        for a2s in (0.05, 0.10, 0.15):
            if a1s + 2*a2s >= 1:
                continue
            res = minimize(neg_loglik, [a0s*m2hat, a1s, a2s], args=(h,),
                           method='SLSQP',
                           bounds=[(1e-12, None), (0, 0.999), (0, 0.5)],
                           constraints=[{'type': 'ineq',
                                         'fun': lambda t: 0.999 - (t[1] + 2*t[2])}])
            if best is None or res.fun < best.fun:
                best = res
a0, a1, a2 = best.x
print("HARCH(2) MLE: a0=%.4e  a1=%.4f  a2=%.4f" % (a0, a1, a2))

# ---------- стандартизованные остатки ----------
sig2 = sigma2(best.x, h)
eps = h / np.sqrt(sig2)

# ---------- критерий операционного времени (Халиуллин) ----------
def khaliullin_homosk(x, m=None):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    if m is None:
        m = n // 4                                    # число уровней ~ N/4
    kappa = np.mean(x**4) / np.mean(x**2)**2          # куртозис (=3 для N(0,1))
    C = np.cumsum(x**2)
    lev = np.arange(1, m+1) * C[-1] / m               # уровни на весь ряд (c = N/(4 sigma^2))
    tau = np.clip(np.searchsorted(C, lev, side='left') + 1, 1, n).astype(float)
    exp = np.arange(1, m+1) * n / m                   # E[tau_theta] при однородности
    stat = np.sum((tau - exp)**2 / exp) / (kappa - 1.0)
    dfree = m - 2
    return stat, dfree, chi2.sf(stat, dfree)

m = N // 4
print("m = N//4 = %d" % m)
st_h, df_h, p_h = khaliullin_homosk(h,   m)   # сами прибыли -> должно отвергаться
st_e, df_e, p_e = khaliullin_homosk(eps, m)   # остатки      -> не должно отвергаться
print("Прибыли  h_n:            chi2=%.1f  df=%d  p=%.3g" % (st_h, df_h, p_h))
print("Остатки  eps=h/sigma:    chi2=%.1f  df=%d  p=%.3f" % (st_e, df_e, p_e))
