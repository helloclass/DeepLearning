import numpy as np
x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y = np.array([-100, -64, -36, -16, -4, 0, 4, 16, 36, 64, 100])

w2_0 = np.random.normal([1])
w2_1 = np.random.normal([1])
b2 = np.random.normal([1])

w3_0 = np.random.normal([1])
w3_1 = np.random.normal([1])
w3_2 = np.random.normal([1])
b3 = np.random.normal([1])

def eta(e):
    return 5 / (100 + e)

for i in range(1000):
    error = np.mean(y - (w2_0 * x * x + w2_1 * x + b2))
    w2_0 = w2_0 + eta(i) * error
    w2_1 = w2_1 + eta(i) * error
    b2 = b2 + eta(i) * error

    error = np.mean(y - (w3_0 * x * x * x + w3_1 * x * x + w3_2 * x + b3))
    w3_0 = w3_0 + eta(i) * error
    w3_1 = w3_1 + eta(i) * error
    w3_2 = w3_2 + eta(i) * error
    b3 = b3 + eta(i) * error

data = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
res2 = w2_0 * data * data + w2_1 * data + b2
res3 = w3_0 * data * data * data + w3_1 * data * data + w3_2 * data + b3

print(res2)
print(res3)

import matplotlib.pyplot as plt
plt.plot(x, y, "r-")
plt.plot(data, res2, "g-")
plt.plot(data, res3, "b-")
plt.show()
