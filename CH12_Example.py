import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
# sequence_loss = tf.contrib.seq2seq.sequence_loss의 contrib 문제를 해결하기 위한
# tfa.seq2seq.sequence_loss를 사용하기 위해  tensorflow_addons를 추가함 (2022.9.14)
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import os
from tf_slim.layers import layers as _layers

# 'gohome'  Data Creation
idx2char = ['g', 'o', 'h', 'm', 'e']  # g = 0, o = 1, h = 2, m = 3, e = 4

x_data = [[0, 1, 2, 1, 3]]   # gohom

x_one_hot = [[[1, 0, 0, 0],   # g 0
              [0, 1, 0, 0],   # o 1
              [0, 0, 1, 0],   # h 2
              [0, 1, 0, 0],   # o 1
              [0, 0, 0, 1]]]  # m 3

y_data = [[1, 2, 1, 3, 4]]    # ohome

num_classes = 5      # 정답 크기, 즉 one-hot 으로 나타내는 크기
input_dim = 4        # one-hot size, 즉 입력값은 0부터 3까지 총 4가지임
hidden_size = 5      # output from the RNN. 5 to directly predict one-hot
batch_size = 1       # one sentence
sequence_length = 5  # 입력으로 들어가는 문장 길이 gohom == 5
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

#cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)  # BasicRNNCell(rnn_size)
cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True) # 초기 상태 지정

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])  # 3-D 구조를 2-D 구조로 변환, hidden_size=5
outputs2 = _layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
outputs3 = tf.reshape(outputs2, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tfa.seq2seq.sequence_loss(logits=outputs3, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):

        loss_val, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(Y, feed_dict={X: x_one_hot})

        if step % 400 == 0:
            print("step = ", step, ", loss = ", loss_val, ", prediction = ", result, ", target = ", y_data)

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)]

            print("\tPrediction = ", ''.join(result_str))

