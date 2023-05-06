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

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 512],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b5 = tf.Variable(tf.random_normal([512]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.get_variable("W6", shape=[512, nb_classes],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b6 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L5, W6) + b6

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
                                                        feed_dict={X: batch_xs, Y: Y_one_hot, keep_prob: 0.7})

            avg_cost += cost_val / total_batch_cnt

            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        if epoch % 1== 0:
           now = datetime.datetime.now()
           nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
           print(nowDatetime,'Epoch:','%07d' % (epoch), 'avg cost=','{:.9f}'.format(avg_cost))


    Y_one_hot = np.eye(nb_classes)[test_labels]

    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess, feed_dict={X: test_images, Y: Y_one_hot, keep_prob: 1}))

