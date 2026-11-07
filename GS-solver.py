import numpy as np

A = np.array([
    [2, 4, -1, 1],
    [3, 4,  0, 2],
    [4, 5,  0, 1],
    [5, 4, -1, 1]
], dtype=float)


B = np.array([229.9, 245.4, 277.8, 276.2], dtype=float)

x = np.linalg.solve(A, B)

print("Solution vector x:")
for i, val in enumerate(x, start=1):
    print(f"x_{i} = {val:.4f}")