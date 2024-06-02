# https://aistudy9314.tistory.com/61
# https://ititit1.tistory.com/61
from __future__ import print_function
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
# sequence_loss = tf.contrib.seq2seq.sequence_loss의 contrib 문제를 해결하기 위한
# tfa.seq2seq.sequence_loss를 사용하기 위해  tensorflow_addons를 추가함 (2022.9.14)
import tensorflow_addons as tfa
import numpy as np
import keras
import matplotlib.pyplot as plt
import random
import datetime
import os
from tf_slim.layers import layers as _layers
# tf.contrib.layers.fully_connected 의 contrib 문제를 해결하기 위한
# _layers.fully_connected를 사용하기 위해  tf_slim.layers 를 추가함 (2022.9.14)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 딕셔너리 생성
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
print('char_set = ', char_set)
print('char_dic = ', char_dic)
'''
char_set =  ['l', 'd', 'i', 'y', 'w', 'b', 'e', 'c', '.', 'r', 'o', 'u', 'n', 'g', 'f', 's', 'k', ',', "'", ' ', 't', 'm', 'p', 'a', 'h']
char_dic =  {'l': 0, 'd': 1, 'i': 2, 'y': 3, 'w': 4, 'b': 5, 'e': 6, 'c': 7, '.': 8, 'r': 9, 'o': 10, 'u': 11, 'n': 12, 'g': 13, 'f': 14, 
             's': 15, 'k': 16, ',': 17, "'": 18, ' ': 19, 't': 20, 'm': 21, 'p': 22, 'a': 23, 'h': 24}
'''

data_dim = len(char_set)
hidden_size = len(char_set)       # input data 차원
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number # 10글자씩 backprop
learning_rate = 0.1

print('data dim =', data_dim)
print('hidden_size =', hidden_size)
print('num_classes =', num_classes)
print('sequence_length =', sequence_length)
print('len(sentence) =', len(sentence))

'''
data dim = 25
hidden_size = 25
num_classes = 25
sequence_length = 10
len(sentence) = 180
'''

# input data 생성
dataX = []
dataY = []
# len(sentence) = 180
for i in range(0, len(sentence) - sequence_length): # 총 180글자를 10글자씩 10글자 전까지 input (0에서 169까지)
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    # 딕셔너리에서 인덱스 불러옴
    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)
    #print('dataX = ', dataX)
    #print('dataY = ', dataY)

batch_size = len(dataX)  # data 크기 만큼

print('dataX = ', dataX)
print('batch_size = ', batch_size)
'''
batch_size =  170
'''

# 10개씩 받아옴으로 10개씩
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
# 정답 One-hot encoding 구현
X_one_hot = tf.one_hot(X, num_classes)
print('X_one_hot= ',X_one_hot)  # check out the shape

# RNN의 개별 셀 생성, hidden_size 차원
# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
#cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=Tru
#rnn 아래와 같이 수정함 (2022.9.14)
     cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
#    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
     return cell

# RNN 셀 중첩, layer_size 층
#multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
multi_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# dynamic_rnn: 일부 학습 반복, 멀티셀, X_one_hot 입력
# outputs: unfolding size x hidden size, state = hidden size
#outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
outputs, _states = tf.nn.dynamic_rnn(multi_cells,X_one_hot, dtype=tf.float32)

# outputs: hidden state 개별 정보 전부 저장되어 있음 , _state: 말단 정보 합쳐져있음

# FC layer
# reshape 로 각 글자 별로 묶음
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# n개 (num_classes) 분류 fully connected layer 자동 구성
outputs = _layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# softmax_w = tf.get_variable("softmax_w",[hidden_size, num_classes])
# softmax_b = tf.get_variable("softmax_b",[num_classes])
# outputs = tf.matmul(X_for_fc, softmax_w) + softmax_b

# reshape out for sequence_loss
# loss 계산 위해 reshape batch_size * input 크기(squence_length) * 정답 개수
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# All weights are 1 (equal weights)
# 가중치 모두 1
weights = tf.ones([batch_size, sequence_length])

# target 별 loss 취합, weights: 특정 character 에 가중치 줌
sequence_loss = tfa.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# outputs: softmax score for each timestamp [[0.1 0.2 0.7], [0.6 0.2 0.2]]
mean_loss = tf.reduce_mean(sequence_loss)
# adam 으로 학습
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

#RNN layer 구성, train, loss 구현

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# 실행
print('========== next character prediction ==========')
# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
       print(''.join([char_set[t] for t in index]), end='')
    else:
       print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616
f you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.
'''