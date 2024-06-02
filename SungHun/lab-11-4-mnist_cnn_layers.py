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

# hyper parameters
nb_classes = 10
learning_rate = 0.001

class Model:

    def __init__(self, sess, name): #세션과 name을 넘겨줘서 초기화를 합니다.
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):   #여기에 모든 레이어를 넣습니다.
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):    # 예측하는 함수
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):    #정확도를 반환합니다.
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):              # 학습하는 함수
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})


# initialize
sess = tf.Session()    # 세션을 생성하고
m1 = Model(sess, "m1")   #세션을 넘겨줘서 m1이라는 모델을 만듭니다.

sess.run(tf.global_variables_initializer())  #tensor를 실행시키기전 초기화를 실행시킵니다.

print('Learning Started!')

num_epochs = 15
batch_size = 30

# train my model
for epoch in range(num_epochs):
    avg_cost = 0
    total_batch_cnt = int(train_images.shape[0] / batch_size)

    for i in range(total_batch_cnt):
        s_idx = int(train_images.shape[0] * i / total_batch_cnt)
        e_idx = int(train_images.shape[0] * (i + 1) / total_batch_cnt)

        batch_xs = train_images[s_idx: e_idx]
        batch_ys = train_labels[s_idx: e_idx]
        Y_one_hot = np.eye(nb_classes)[batch_ys] #y값 2을 0010000000 형태로 변경

        c, _ = m1.train(batch_xs, Y_one_hot)  # 데이터를 가지고 train을 호출합니다.

        avg_cost += c / total_batch_cnt

    if epoch % 1 == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime, 'Epoch:', '%07d' % (epoch), 'avg cost=',
              '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
Y1_one_hot = np.eye(nb_classes)[test_labels]
print('Accuracy:', m1.get_accuracy(test_images, Y1_one_hot))

