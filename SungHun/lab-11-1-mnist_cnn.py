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
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, nb_classes])

print('###################################################################')
print('MNIST Convolution Layer 1 ')
print('###################################################################')

# 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
# W1 [3 3 1 32] -> [3 3]: 필터 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
# 그리고 X_img = tf.reshape(X, [-1, 28, 28, 1]) 의 1과
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))의 1이 같다.

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)  <- Padding same 사용해서 필터크기(3*3)에 상관 없이
#                                    입력, 출력 이미지 크기가 같음
#    Pool     -> (?, 14, 14, 32)
# tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
# padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션이다.
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

print('###################################################################')
print('MNIST Max Pooling 1')
print('###################################################################')

# Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
# 2*2 크기로 2칸씩 움직여 14*14 크기로 변경됨
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''
print('###################################################################')
print('MNIST Convolution Layer 2 ')
print('###################################################################')

# 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
# W1 [3 3 1 64] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 64: 필터 갯수
# W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
# tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
# padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

print('###################################################################')
print('MNIST Max Pooling 2')
print('###################################################################')

# Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
# 2*2 크기로 2칸씩 움직여 7*7 크기로 변경됨
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

print('###################################################################')
print(' Fully Connected (FC, Dense) layer')
print('###################################################################')
# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, nb_classes],
                     initializer = tf.truncated_normal_initializer(stddev=0.1))
b = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print('###################################################################')
print(' Training and Evalulation ')
print('###################################################################')

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

            _, cost_val, y_val, h_val \
                = sess.run([train, cost, Y, logits], feed_dict={X: batch_xs, Y: Y_one_hot})

            avg_cost += cost_val / total_batch_cnt

        if epoch % 1== 0:
           now = datetime.datetime.now()
           nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
           print(nowDatetime,'Epoch:','%07d' % (epoch), 'avg cost=','{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    print('###################################################################')
    print(' Test Model and check accuracy ')
    print('###################################################################')

    prediction = tf.argmax(logits, 1)
    is_correct = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy   = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    Y_one_hot  = np.eye(nb_classes)[test_labels]

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess, feed_dict={X: test_images, Y: Y_one_hot}))

