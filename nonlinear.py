from scipy.optimize import minimize
import numpy as np
np.set_printoptions(suppress=True)

def f(x):
    k, q, r = x
    S_u = 0.045
    S_b = r
    return abs(((k*q - 1) * S_u * S_b) / (S_u + k*S_b))

def ftest(k, q, r):
    S_u = 0.045
    S_b = r
    return ((k*q - 1) * S_u * S_b) / (S_u + k*S_b)

bounds = [(0, 21.5), (0.5, 1), (0.5, 1)]
res = minimize(f, x0=[1.5, 1, 1], bounds=bounds, method='SLSQP')
print(res)

answer = res.x

k = round(answer[0], 4)
q = round(answer[1], 4) + 0.01
r = round(answer[2], 4)

print("k = {:.4f}".format(k))
print("q = {:.4f}".format(q))
print("r = {:.4f}".format(r))
print("E(B) = {:.20f}".format(ftest(k, q, r)))
