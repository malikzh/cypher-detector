def f_test(k,q,r):
    return (k*q-1)*r/(1+k*r)

q = 0.77
r = 1.08

for k in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    E_B = round(f_test(k, q, r), 4)
    P_BQ = round(r / (1 + k*r), 4)
    P_B = round(q * P_BQ, 4)
    print(f"{k} & {P_B} & {E_B} \\\\")

exit()
from scipy.optimize import minimize
import numpy as np
np.set_printoptions(suppress=True)

def f(x):
    q, r = x
    return abs((1.5*q-1)*r/(1+1.5*r))

def f_nonabs(x):
    q, r = x
    return (1.5*q-1)*r/(1+1.5*r)

bounds = [(0.49, 0.85), (0.9, 3.4)]
res = minimize(f, x0=[0.5, 1.0], bounds=bounds, method='SLSQP')
print(res)

q = round(res.x[0] + 0.1, 2)
r = round(res.x[1] + 0.1, 2)
print("Result: {:.20f}".format(f_nonabs([q, r])))
print("Numbers: q={:.2f}, r={:.2f}".format(q, r))