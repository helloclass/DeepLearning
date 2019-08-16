import numpy as np
import csv
learning_rate = 0.05

def abalone_exec(epoch):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch)

def load_abalone_dataset():
    data_set = open("abalone.csv")
    data_reader = csv.reader(data_set)
    next(data_reader, None)
    rows = []
    for i in data_reader:
        rows.append(i)
    global data_x, data_y, data_size, x_size, y_size
    data_size, x_size, y_size = len(rows), 10, 1
    data_x=np.zeros([data_size, 10])
    data_y=np.zeros([data_size])
    for i, a in enumerate(rows):
        if(a[0] == 'I'): data_x[i][0]=1
        elif(a[0] == 'M'): data_x[i][1]=1
        elif(a[0] == 'F'): data_x[i][2]=1
        data_x[i][3:] = a[1:-1]
        data_y[i] = a[-1]

def init_model():
    global W, b, x_size, y_size
    W = np.random.normal(0, 1, [x_size, y_size]) #(10, 1)
    b = np.random.normal([y_size])
    
def train_and_test(epoch):
    for i in range(epoch):
        train_x, train_y = get_train_data()
        for n in range(train_x.shape[0]):
            run_train(train_x[n], train_y[n])
    test_x, test_y = get_test_data()
    cost = 0.0
    for n in range(test_x.shape[0]):
        loss = run_test(test_x[n], test_y[n])
        cost += loss
    print("result cost: ", cost / test_x.shape[0])
    
def arange_data():
    global data_x, data_y, data_size, x_size, y_size, test_start_idx
    test_start_idx = int(data_size * 0.8)
    train_data_x, test_data_x = data_x[:test_start_idx], data_x[test_start_idx:]
    train_data_y, test_data_y = data_y[:test_start_idx], data_y[test_start_idx:]
    return train_data_x, train_data_y, test_data_x, test_data_y

def get_train_data():
    train_data_x, train_data_y, _, _ = arange_data()
    return train_data_x, train_data_y

def get_test_data():
    _, _, test_data_x, test_data_y = arange_data()
    return test_data_x, test_data_y

def run_train(x, y):
    z = forward_neuralnet(x)
    forward_postproc(z, y)
    d = backprop_neuralnet(z, y)
    backprop_postproc(d, x)
    
def run_test(x, y):
    z = forward_neuralnet(x)
    forward_postproc(z, y)
    cost = eval_accuracy(z, y)
    return cost
    
def forward_neuralnet(x):
    global W, b
    x = np.reshape(x, [1, x.shape[0]])
    z = np.matmul(x, W) + b
    return z
    
def forward_postproc(z, y):
    a = np.mean(np.power(z - y, 2))
    return a
    
def eval_accuracy(z, y):
    cost = np.sqrt(np.mean(np.power(z - y, 2)))
    return cost
    
def backprop_neuralnet(z, y):
    global x_size, y_size
    z = np.reshape(z, [z.shape[0]])
    d_1 = 2 / (x_size * y_size)
    d = d_1 * (z - y)
    return d
    
def backprop_postproc(d, x):
    global learning_rate, W, b
    d_w = d * x * learning_rate
    d_b = d * learning_rate
    d_w = np.reshape(d_w, [d_w.shape[0], 1])
    
    W = W - d_w
    b = b - d_b
    
abalone_exec(100)
