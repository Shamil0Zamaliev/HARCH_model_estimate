import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time

def simulate(a0,a1,a2,N,burn,rng):
    h=np.zeros(N+burn+2); eps=rng.standard_normal(N+burn+2)
    for n in range(2,N+burn+2):
        s2=a0+a1*h[n-1]**2+a2*(h[n-1]+h[n-2])**2
        h[n]=np.sqrt(max(s2,1e-12))*eps[n]
    return h[burn+2:]

def m2_exists(a1,a2): return a1+2*a2<1

def m4_exists(a1,a2):
    if a2>=1: return False
    return 3*((a1+a2)**2+a2**2)+a2*(1+3*(a1**2+6*a1*a2+4*a2**2))<1

def a2_m4(a1):
    lo,hi=0.0,(1-a1)/2-1e-6
    if hi<=0: return 0.0
    for _ in range(40):
        m=(lo+hi)/2
        if m4_exists(a1,m): lo=m
        else: hi=m
    return lo

def L4_matrix(a1, a2):
    return np.array([
        [105*(a1**4 + 4*a1**3*a2 + 6*a1**2*a2**2 + 4*a1*a2**3 + 2*a2**4),
         420*a2*(a1+a2)**2*(a1+7*a2),
         210*a2**2*(3*a1**2 + 30*a1*a2 + 35*a2**2),
         420*a2**3*(a1+7*a2)],
        [15*(a1+a2)**3, 45*a2*(a1+a2)*(a1+5*a2), 45*a2**2*(a1+5*a2), 15*a2**3],
        [3*(a1+a2)**2, 6*a2*(a1+3*a2), 3*a2**2, 0.0],
        [a1+a2, a2, 0.0, 0.0]
    ])

def m8_factor(a1, a2):
    # det(I - L4) > 0  <=>  m8 конечен
    return np.linalg.det(np.eye(4) - L4_matrix(a1, a2))

def a2_m8(a1):
    """Аналитическая граница m8: максимальное a2 при данном a1, где det(I-L4)>0."""
    top = a2_m4(a1)
    if top <= 1e-6:
        return 0.0
    # при a2=0 (ARCH(1)) условие 105*a1^4 < 1
    if m8_factor(a1, 0.0) <= 0:
        return 0.0
    lo, hi = 0.0, top
    for _ in range(60):          # побольше итераций для точности
        mid = (lo + hi) / 2
        if m8_factor(a1, mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo

# ========== ОСНОВНОЙ БЛОК ==========
A0 = 0.05
a1_axis = np.linspace(0.001, 0.49, 100)   # достаточно точек для гладкой кривой

t0 = time.time()
b_m8 = np.array([a2_m8(a) for a in a1_axis])
print(f"Аналитическая граница m8 посчитана за {time.time()-t0:.0f} с")

# --- Точные кривые для m2 и m4 ---
a1_fine = np.linspace(0, 0.99, 500)
m2_curve = (1 - a1_fine) / 2
m4_curve = np.array([a2_m4(a) for a in a1_fine])

# --- Построение карты областей ---
fig, ax = plt.subplots(figsize=(8, 6))
A1, A2 = np.meshgrid(np.linspace(0, 0.6, 400), np.linspace(0, 0.30, 400))
reg = np.zeros_like(A1, dtype=int)

for i in range(A1.shape[0]):
    for j in range(A1.shape[1]):
        a1, a2 = A1[i, j], A2[i, j]
        if not m2_exists(a1, a2):
            reg[i, j] = 0
        elif not m4_exists(a1, a2):
            reg[i, j] = 1
        elif a2 > np.interp(a1, a1_axis, b_m8, right=0.0):
            reg[i, j] = 2
        else:
            reg[i, j] = 3

colors = ['#f0f0f0', '#d6c4e0', '#f6c89a', '#7fc97f']
ax.pcolormesh(A1, A2, reg, cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5, shading='auto')

# Кривые границ
ax.plot(a1_fine, m2_curve, 'k-', lw=2, label=r'$a_1+2a_2=1$ (граница $m_2$)')
mask4 = m4_curve > 1e-4
ax.plot(a1_fine[mask4], m4_curve[mask4], 'r--', lw=2, label=r'граница $m_4<\infty$ (ММ состоятелен)')
mask8 = b_m8 > 1e-4
ax.plot(a1_axis[mask8], b_m8[mask8], 'b-', lw=2.5, marker='o', ms=4,
        label=r'граница $m_8<\infty$ (ММ имеет конечную Var)')

ax.set_xlabel(r'$a_1$', fontsize=13)
ax.set_ylabel(r'$a_2$', fontsize=13)
ax.set_title(r'Области существования моментов HARCH(2) ($a_0=0.05$)', fontsize=12)
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.30)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regions_m2_m4_m8.png', dpi=150)
print("Сохранено: regions_m2_m4_m8.png")

# Вывод значений для справки
print("a1, граница a2(m8):")
for a, b in zip(a1_axis, b_m8):
    print(f"  a1={a:.3f}  a2_m8={b:.4f}  a2_m4={a2_m4(a):.4f}")