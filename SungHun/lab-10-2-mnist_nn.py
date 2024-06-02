import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape([-1, 784])
test_images  = test_images.reshape([-1, 784])

train_images = train_images / 255.
test_images  = test_images  / 255.

# parameters
nb_classes = 10
learning_rate = 0.001

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=hypothesis, labels=tf.stop_gradient(Y)
    )
)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)

is_correct = tf.equal(prediction, tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 15
batch_size = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        # train_images.shape[0] (60,000) / batch_size로 나눈다.
        total_batch_cnt = int(train_images.shape[0] / batch_size)

        for i in range(total_batch_cnt):
            s_idx = int(train_images.shape[0] * i / total_batch_cnt)
            e_idx = int(train_images.shape[0] * (i + 1) / total_batch_cnt)

            batch_xs = train_images[s_idx: e_idx]
            batch_ys = train_labels[s_idx: e_idx]

            Y_one_hot = np.eye(nb_classes)[batch_ys]

            _, cost_val, y_val, h_val, p_val = sess.run([train, cost, Y, hypothesis, prediction],
                                                        feed_dict={X: batch_xs, Y: Y_one_hot})

            avg_cost += cost_val / total_batch_cnt

            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        if epoch % 10== 0:
           now = datetime.datetime.now()
           nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
           print(nowDatetime,'Epoch:','%07d' % (epoch), 'avg cost=','{:.9f}'.format(avg_cost),
                             '\nhypothesis[0]:\n',h_val[0],'\nprediction[0]:\n',p_val[0])


    Y_one_hot = np.eye(nb_classes)[test_labels]

    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess, feed_dict={X: test_images, Y: Y_one_hot}))

