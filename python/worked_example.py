# worked_example.py
# The small five-house example from the report, done step by step with numpy
# so you can check every number we worked out by hand.

import numpy as np

x = np.array([40, 55, 60, 75, 90], dtype=float)      # floor area (m^2)
y = np.array([18, 22, 25, 28, 34], dtype=float)      # rent (KES '000)
n = len(x)

x_bar, y_bar = x.mean(), y.mean()
Sxx = np.sum((x - x_bar) ** 2)
Sxy = np.sum((x - x_bar) * (y - y_bar))
Syy = np.sum((y - y_bar) ** 2)

b1 = Sxy / Sxx
b0 = y_bar - b1 * x_bar
y_hat = b0 + b1 * x
resid = y - y_hat
SSE = np.sum(resid ** 2)
R2 = 1 - SSE / Syy
s = np.sqrt(SSE / (n - 2))
se_b1 = s / np.sqrt(Sxx)
t_b1 = b1 / se_b1

print(f"x_bar = {x_bar:.2f},  y_bar = {y_bar:.2f}")
print(f"Sxx = {Sxx:.2f},  Sxy = {Sxy:.2f},  Syy = {Syy:.2f}")
print(f"b1 = {b1:.4f},  b0 = {b0:.4f}")
print("fitted:", np.round(y_hat, 3))
print("residuals:", np.round(resid, 3))
print(f"SSE = {SSE:.4f},  R^2 = {R2:.4f},  s = {s:.4f}")
print(f"SE(b1) = {se_b1:.4f},  t = {t_b1:.3f}")

# the same answer through the matrix formula  beta = (X'X)^-1 X'y
X = np.column_stack([np.ones(n), x])
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("matrix solution:", np.round(beta, 4))
