import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
batch_size = 256

def autoEncoder(X):
    W0 = tf.Variable(tf.random.normal(shape=[784, 256]))
    b0 = tf.Variable(tf.random.normal(shape=[256]))
    z0 = tf.matmul(X, W0) + b0
    a0 = tf.nn.sigmoid(z0)

    W1 = tf.Variable(tf.random.normal(shape=[256, 128]))
    b1 = tf.Variable(tf.random.normal(shape=[128]))
    z1 = tf.matmul(a0, W1) + b1
    a1 = tf.nn.sigmoid(z1)

    W2 = tf.Variable(tf.random.normal(shape=[128, 256]))
    b2 = tf.Variable(tf.random.normal(shape=[256]))
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)

    W3 = tf.Variable(tf.random.normal(shape=[256, 784]))
    b3 = tf.Variable(tf.random.normal(shape=[784]))
    z3 = tf.matmul(a2, W3) + b3
    a3 = tf.nn.sigmoid(z3)

    return a3

def ANN(X):
    W0 = tf.Variable(tf.random.normal(shape=[784, 256]))
    b0 = tf.Variable(tf.random.normal(shape=[256]))
    z0 = tf.matmul(X, W0) + b0
    a0 = tf.nn.sigmoid(z0)

    W1 = tf.Variable(tf.random.normal(shape=[256, 256]))
    b1 = tf.Variable(tf.random.normal(shape=[256]))
    z1 = tf.matmul(a0, W1) + b1
    a1 = tf.nn.sigmoid(z1)

    W2 = tf.Variable(tf.random.normal(shape=[256, 10]))
    b2 = tf.Variable(tf.random.normal(shape=[10]))
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)

    return z2, a2

reconstructed_img = autoEncoder(X)
y_true = X

reconstructed_loss = tf.reduce_mean(tf.pow(reconstructed_img - y_true, 2))
reconstructed_train = tf.train.RMSPropOptimizer(0.02).minimize(reconstructed_loss)

logits, y_pred = ANN(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
train = tf.train.AdamOptimizer(0.002).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(30):
    bundle = int(mnist.train.num_examples / batch_size)
    for i in range(bundle):
        data_x, data_y = mnist.train.next_batch(batch_size)
        _, rc_cost = sess.run([reconstructed_train, reconstructed_loss], feed_dict={X: data_x})
    print("epoch: ", epoch, " reconstructed loss rate: ", rc_cost)

reconstructed_predict = sess.run(reconstructed_img, feed_dict={X: mnist.test.images})
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(reconstructed_predict[i], (28, 28)))

f.savefig("res.png")

for epoch in range(30):
    bundle = int(mnist.train.num_examples / batch_size)
    for i in range(bundle):
        data_x, data_y = mnist.train.next_batch(batch_size)
        _, cost = sess.run([train, loss], feed_dict={X: data_x, y: data_y})
    print("epoch: ", epoch, " loss rate: ", cost)


