import numpy as np
import matplotlib.pyplot as plt

data_1 = np.array([[1.0, 1.2, 1.1, 1.4, 1.5, 0.6, 1.8, 1.6, 2.5], [0.5, 1.4, 1.2, 0.8, 1.1, 1.4, 1.3, 1.6, 2.0]])
data_2 = np.array([[2.3, 2.5, 2.7, 2, 2.2, 2.9, 3.0, 3.2, 1.7], [2.9, 2.0, 3.4, 3.0, 3.6, 2.7, 3.1, 2.1, 1.6]])
data_1 = data_1 - 3
data_2 = data_2 - 2

plt.plot(data_1[0], data_1[1], "r.")
plt.plot(data_2[0], data_2[1], "bx")

dist = 100
now_d = 0
index_short = np.array([0, 0])

for i in range(data_1.shape[1]):
    for j in range(data_2.shape[1]):
        now_d = np.sqrt(np.power(data_1[0][i] - data_2[0][j], 2) + np.power(data_1[1][i] - data_2[1][j], 2))
        if dist > now_d:
            dist = now_d
            index_short[0] = i
            index_short[1] = j

res_mat = np.array([[data_1[0][index_short[0]], data_2[0][index_short[1]]], [data_1[1][index_short[0]], data_2[1][index_short[1]]]])
plt.plot([res_mat[0][0], res_mat[0][1]], [res_mat[1][0], res_mat[1][1]], "g-")
mean_x = (res_mat[0][1] - res_mat[0][0]) / 2 + res_mat[0][0]


perp_grad = (res_mat[1][1] - res_mat[1][0]) / (res_mat[0][1] - res_mat[0][0])
perp_grad = -1 / perp_grad
plt.plot([-5 + mean_x, 5 + mean_x], [-5 * perp_grad + mean_x, 5 * perp_grad + mean_x], "g--")

plt.show()
