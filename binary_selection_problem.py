import numpy as np
import csv
learning_rate = 0.07

def pulsar_exec(epoch):
    load_pulsar_datasets()
    init_model()
    train_and_test(epoch)
    
def load_pulsar_datasets():
    data_set = open("Downloads/source/Academy/data/chap02/pulsar_stars.csv")
    data_reader = csv.reader(data_set)
    next(data_reader, None)
    rows = []
    for row in data_reader:
        rows.append(row)
    global data_x, data_y, data_size, x_size, y_size
    data_size, x_size, y_size = len(rows), 8, 1
    data_x = np.zeros([data_size, x_size])
    data_y = np.zeros([data_size, y_size])
    for i, a in enumerate(rows):
        data_x[i] = a[:-1]
        data_y[i] = a[8]

def init_model():
    global W, b, x_size, y_size
    W = np.random.normal(0, 1, [x_size, y_size])
    b = np.random.normal(0, 1, [y_size])
    
def train_and_test(epoch):
    global data_size
    loading = 0
    for i in range(epoch):
        data_train_x, data_train_y = get_train_data()
        for idx in range(len(data_train_x)):
            run_train(data_train_x[idx], data_train_y[idx])
        loading = i / epoch * 100
        print("now epoch: ", i)
    data_test_x, data_test_y = get_test_data()
    cost = 0
    one_answer_num = 0
    for idx in range(len(data_test_x)):
            a, b = run_test(data_test_x[idx], data_test_y[idx])
            cost += a
            one_answer_num += b
    print("result cost(NUMBER that correct one type answer): ", cost , "/", one_answer_num)
    print("acuracy: {0}%".format((1 - (cost / data_test_x.shape[0]))*100))
    
def arrange_data():
    global data_x, data_y, data_size, start_train_idx
    start_train_idx = int(data_size * 0.8)
    data_train_x, data_test_x = data_x[:start_train_idx], data_x[start_train_idx:]
    data_train_y, data_test_y = data_y[:start_train_idx], data_y[start_train_idx:]
    return data_train_x, data_train_y, data_test_x, data_test_y

def get_train_data():
    data_x, data_y, _, _ = arrange_data()
    return data_x, data_y

def get_test_data():
    _, _, data_x, data_y = arrange_data()
    return data_x, data_y

def run_train(x, y):
    z = forward_neuralnet(x)
    a = forward_postproc(z, y)
    d = backprop_neuralnet(z, y)
    backprop_postproc(d, x)
    
def run_test(x, y):
    z = forward_neuralnet(x)
    a = forward_postproc(z, y)
    cost = eval_accuracy(z, y)
    return cost
    
def sigmoid_cross_entropy_with_logits(x, z):
    return x - x*z + np.log(1 + np.exp(-x))

def sigmoid_cross_entropy_with_logits_dervs(x, z):
    return -z + sigmoid(-x)

def relu(x):
    return np.max(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_neuralnet(x):
    global W, b
    x = np.reshape(x, [1, x.shape[0]])
    z = np.matmul(x, W) + b
    return z
    
def forward_postproc(z, y):
    a = relu(z) - z * y + np.log(1  + np.exp(-np.abs(z)))
    return a
    
def eval_accuracy(z, y):
    r_z = sigmoid(z)
    result_z = int(sigmoid(z))
    if result_z == int(y[0]) and y == 1:
        return 1, 1
    elif result_z != int(y[0]) and y == 1:
        return 0, 1
    else:
        return 0, 0
    
def backprop_neuralnet(z, y):
    return -y + sigmoid(z)
    
def backprop_postproc(d, x):
    global W, b, learning_rate
    x_t = np.reshape(x, [8, 1])
    d_W = d * learning_rate * x_t
    d_b = d * learning_rate
    d_b = np.reshape(d_b, [1])
    
    W -= d_W
    b -= d_b
    
for i in range(30):
    pulsar_exec(10)
