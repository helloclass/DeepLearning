import numpy as np
data = np.array([0, 0, 1, 1, 0, 0,
                 0, 1, 0, 0, 1, 0,
                 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 1, 0, 1,
                 0, 1, 0, 0, 1, 0,
                 0, 0, 1, 1, 0, 0])
target = np.array([1, 0])

F_arr = np.array([[0, 0, 1,
                   0, 1, 0,
                   1, 0, 0],
                  [0, 1, 0,
                   0, 1, 0,
                   0, 1, 0],
                  [1, 0, 0,
                   0, 1, 0,
                   0, 0, 1],
                  [0, 0, 0,
                   1, 1, 1,
                   0, 0, 0]])

CNN = np.array([[0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0],
                [0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0]
                ])

def sigmoid(a):
    return 1/(1+np.exp(-a))

def cal_CNN(CNN_num, data_n, F_n):
    iter = data_n - F_n + 1
    for epoch in range(CNN_num):
        for i in range(iter):
            x = i
            for j in range(iter):
                y = j
                for x in range(F_n):
                    for y in range(F_n):
                        CNN[epoch][i*iter+j] = CNN[epoch][i*iter+j]+data[x*data_n+y+i*data_n+j] * F_arr[epoch][y*F_n+x]

    return CNN

cal_CNN(4, 6, 3)

CNN_b = np.random.normal(0, 1, (1, 16))
for i in range(4):
    CNN[i] = CNN[i] + CNN_b

CNN_a = sigmoid(CNN)

POOLING = np.array([[0.0, 0.0,
                     0.0, 0.0],
                    [0.0, 0.0,
                     0.0, 0.0],
                    [0.0, 0.0,
                     0.0, 0.0],
                    [0.0, 0.0,
                     0.0, 0.0]])

def cal_pool(pool_num, num_CNN):
    half_num = int(num_CNN/2)
    for epoch in range(pool_num):
        for i in range(half_num):
            for j in range(half_num):
                arr = np.array([CNN_a[epoch][i*2*num_CNN + j*2], CNN_a[epoch][i*2*num_CNN + j*2+1], CNN_a[epoch][(i*2+1)*num_CNN + j*2], CNN_a[epoch][(i*2+1)*num_CNN + j*2+1]])
                POOLING[epoch][i*half_num+j] = np.amax(arr)
    return POOLING

POOLING = cal_pool(4, 4)
POOLING = np.reshape(POOLING, (1, 16))

W_o = np.random.normal(0, 1, (16, 3))
b_o = np.random.normal(0, 1, (1, 3))

z_o = POOLING.dot(W_o) + b_o
z_o = np.reshape(z_o, (3))
print(z_o.shape)

C_t = 1/2*(np.power(z_o[0], 2) + np.power(z_o[1], 2) + np.power(z_o[2], 2))

print("res: ", np.argmax(z_o))
print("z_o: ", z_o[0], z_o[1], z_o[2])
print("cost: ", C_t)

#다음에는 CNN을 backpropagation을 이용하여 학습 시켜.
