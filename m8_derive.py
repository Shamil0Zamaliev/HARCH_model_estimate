# -*- coding: utf-8 -*-
"""Аналитический (символьный) вывод m6, m8 для HARCH(2) и условий их конечности.
Метод: m_{2k} = (2k-1)!! E[sigma_n^{2k}]; приведение к линейной системе
стационарных чётно-чётных моментов M(p,q)=E[h_n^{2p} h_{n-1}^{2q}] через
условное усреднение. Условие конечности m_{2k} = спектральный радиус
матрицы переноса верхнего порядка < 1."""
import sympy as sp
from sympy import factorial2

a0, a1, a2 = sp.symbols('a0 a1 a2', positive=True)
X, Y = sp.symbols('X Y')
b = a1 + a2
q = a0 + b*X**2 + 2*a2*X*Y + a2*Y**2     # sigma_n^2 как квадратичная форма от (h_{n-1},h_{n-2})

m2, m4, R1, m6, S21, S12, m8, T31, T22, T13 = sp.symbols(
    'm2 m4 R1 m6 S21 S12 m8 T31 T22 T13')

table = {(0,0): sp.Integer(1),
         (1,0): m2, (0,1): m2,
         (2,0): m4, (0,2): m4, (1,1): R1,
         (3,0): m6, (0,3): m6, (2,1): S21, (1,2): S12,
         (4,0): m8, (0,4): m8, (3,1): T31, (1,3): T13, (2,2): T22}

def Emono(i, j):
    if i % 2 == 1 or j % 2 == 1:
        return sp.Integer(0)
    return table[(i//2, j//2)]

def dfact(p):
    return factorial2(2*p-1)

targets = [(1,0),(2,0),(1,1),(3,0),(2,1),(1,2),(4,0),(3,1),(2,2),(1,3)]
unknowns = [m2, m4, R1, m6, S21, S12, m8, T31, T22, T13]

eqs = []
rhs_by_target = {}
for (p, qq) in targets:
    expr = sp.expand((q**p) * X**(2*qq))
    poly = sp.Poly(expr, X, Y)
    rhs = sp.Integer(0)
    for pows, coeff in poly.terms():
        i, j = pows
        rhs += coeff*Emono(i, j)
    rhs = sp.expand(dfact(p)*rhs)
    rhs_by_target[(p, qq)] = rhs
    eqs.append(sp.Eq(table[(p, qq)], rhs))

sol = sp.solve(eqs, unknowns, dict=True)[0]

m2v = sp.simplify(sol[m2])
m4v = sp.simplify(sol[m4])
m6v = sp.simplify(sol[m6])
m8v = sp.simplify(sol[m8])

print("=== m2 ==="); sp.pprint(sp.factor(m2v))
print("\n=== m4 (denominator) ===")
print(sp.factor(sp.denom(sp.cancel(m4v))))
print("  a2=0 ->", sp.factor(sp.denom(sp.cancel(m4v)).subs(a2,0)))

print("\n=== m6 (denominator factored) ===")
den6 = sp.factor(sp.denom(sp.cancel(sp.together(m6v))))
print(den6)
print("  a2=0 ->", sp.factor(den6.subs(a2,0)))

print("\n=== m8 (denominator factored) ===")
den8 = sp.factor(sp.denom(sp.cancel(sp.together(m8v))))
print(den8)
print("  a2=0 ->", sp.factor(den8.subs(a2,0)))

# --- матрица переноса 4-го порядка L4 на (m8,T31,T22,T13) ---
ord4 = [m8, T31, T22, T13]
ord4_targets = [(4,0),(3,1),(2,2),(1,3)]
L4 = sp.zeros(4,4)
for r,(p,qq) in enumerate(ord4_targets):
    rhs = rhs_by_target[(p,qq)]
    for c,sym in enumerate(ord4):
        L4[r,c] = sp.expand(rhs).coeff(sym)
print("\n=== L4 (матрица переноса 4-го порядка) ===")
sp.pprint(L4)
ImL4 = sp.eye(4) - L4
detv = sp.factor(sp.det(ImL4))
print("\n=== det(I - L4) (фактор условия m8<inf) ===")
print(detv)
print("  a2=0 ->", sp.factor(detv.subs(a2,0)))

print("\n=== L4 поэлементно (факторизовано) ===")
for r in range(4):
    for c in range(4):
        print(f"L4[{r+1},{c+1}] =", sp.factor(L4[r,c]))

# m4-условие из Ch.1 для сверки
cond_m4_ch1 = 3*(a2**2+(a1+a2)**2)+a2*(1+3*(a1**2+6*a1*a2+4*a2**2))-1
print("\n=== сверка m4-условия с гл.1 (должно быть 0) ===")
print(sp.simplify(sp.factor(sp.denom(sp.cancel(m4v))/(a1+2*a2-1)) - cond_m4_ch1))

# Численная граница m8 при нескольких a2 (где det(I-L4)=0 по a1)
print("\n=== численная граница a1*(a2): m8<inf  <=>  a1 < a1*  ===")
import numpy as np
f = sp.lambdify((a1,a2), detv, 'numpy')
from scipy.optimize import brentq
for a2v in [0.0,0.02,0.05,0.08,0.10,0.12,0.15]:
    try:
        root = brentq(lambda x: float(f(x,a2v)), 1e-6, 0.6)
        print(f"  a2={a2v:.2f}:  a1*(m8) = {root:.4f}")
    except Exception as e:
        print(f"  a2={a2v:.2f}:  нет корня в (0,0.6)")
