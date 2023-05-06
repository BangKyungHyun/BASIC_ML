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

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sample = " if you want you"
unique_char = set(sample)          # 먼저 unique한 알파벳 만을 추려낸다
idx2char = list(set(unique_char))  # 추출된 결과는 set이므로 이를 list로 만든다.
char2idx = {c: i for i, c in enumerate(idx2char)}  # 인덱스 값을 Value로, 해당 인덱스의 알파벳을 Key로 구성된 데이터는 다음처럼 얻을 수 있다.
print('unique_char =', unique_char)
print('idx2char =', idx2char)
print('enumerate(idx2char) =', enumerate(idx2char))
print('char2idx =', char2idx)
'''
unique_char = {'i', 'o', ' ', 'u', 't', 'n', 'a', 'w', 'f', 'y'}
idx2char = ['i', 'o', ' ', 'u', 't', 'n', 'a', 'w', 'f', 'y']
enumerate(idx2char) = <enumerate object at 0x0000025C8E625340>
char2idx = {'i': 0, 'o': 1, ' ': 2, 'u': 3, 't': 4, 'n': 5, 'a': 6, 'w': 7, 'f': 8, 'y': 9}
'''
# hyper parameters
dic_size    = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size  = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

print('dic size =', dic_size)
print('hidden_size =', hidden_size)
print('num_classes =', num_classes)
print('sequence_length =', sequence_length)
'''
dic size = 10
hidden_size = 10
num_classes = 10
sequence_length = 15
'''
sample_idx = [char2idx[c] for c in sample]  # char to index

print('sample_idx =',sample_idx)
'''
sample_idx = [2, 0, 8, 2, 9, 1, 3, 2, 7, 6, 5, 4, 2, 9, 1, 3]
'''
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

print('x_data =',x_data)
print('y_data =',y_data)
'''
x_data = [[2, 0, 8, 2, 9, 1, 3, 2, 7, 6, 5, 4, 2, 9, 1]]
y_data = [[0, 8, 2, 9, 1, 3, 2, 7, 6, 5, 4, 2, 9, 1, 3]]
'''

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
#cell = tf.contrib.rnn.BasicLSTMCell(
## contrib 아래와 같이 수정함 (2022.9.14)
cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
## contrib 아래와 같이 수정함 (2022.9.14)
outputs = tf.compat.v1.layers.dense(X_for_fc, num_classes, activation=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])

#sequence_loss = tf.contrib.seq2seq.sequence_loss(
## contrib 아래와 같이 수정함 (2022.9.14)
sequence_loss = tfa.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))



'''
1 loss: 2.1711426 Prediction: y   ou  oooooou
2 loss: 2.0561042 Prediction: y    u         
3 loss: 1.9322335 Prediction: y  y u         
4 loss: 1.771687 Prediction: y  you yano you
5 loss: 1.5269151 Prediction: yy you yano you
6 loss: 1.3053467 Prediction: yf you yano you
7 loss: 1.0064555 Prediction: yf you uant you
8 loss: 0.7446786 Prediction: if you want you
9 loss: 0.5246001 Prediction: if you want you
10 loss: 0.35779542 Prediction: if you want you
11 loss: 0.23477599 Prediction: if you want you
12 loss: 0.15142158 Prediction: if you want you
13 loss: 0.09785592 Prediction: if you want you
14 loss: 0.06463547 Prediction: if you want you
15 loss: 0.04404514 Prediction: if you want you
16 loss: 0.030522177 Prediction: if you want you
17 loss: 0.021431098 Prediction: if you want you
18 loss: 0.015368565 Prediction: if you want you
19 loss: 0.011304065 Prediction: if you want you
20 loss: 0.00852511 Prediction: if you want you
21 loss: 0.0065835393 Prediction: if you want you
22 loss: 0.0052011493 Prediction: if you want you
23 loss: 0.004199706 Prediction: if you want you
24 loss: 0.003460788 Prediction: if you want you
25 loss: 0.00290481 Prediction: if you want you
26 loss: 0.0024779146 Prediction: if you want you
27 loss: 0.002143552 Prediction: if you want you
28 loss: 0.0018767581 Prediction: if you want you
29 loss: 0.0016603334 Prediction: if you want you
30 loss: 0.0014822894 Prediction: if you want you
31 loss: 0.001334065 Prediction: if you want you
32 loss: 0.0012094055 Prediction: if you want you
33 loss: 0.0011037027 Prediction: if you want you
34 loss: 0.001013432 Prediction: if you want you
35 loss: 0.0009359318 Prediction: if you want you
36 loss: 0.0008689203 Prediction: if you want you
37 loss: 0.00081089255 Prediction: if you want you
38 loss: 0.0007602396 Prediction: if you want you
39 loss: 0.00071601063 Prediction: if you want you
40 loss: 0.0006771991 Prediction: if you want you
41 loss: 0.00064297265 Prediction: if you want you
42 loss: 0.0006127208 Prediction: if you want you
43 loss: 0.00058591255 Prediction: if you want you
44 loss: 0.0005620083 Prediction: if you want you
45 loss: 0.0005407229 Prediction: if you want you
46 loss: 0.0005216357 Prediction: if you want you
47 loss: 0.0005044852 Prediction: if you want you
48 loss: 0.00048907293 Prediction: if you want you
49 loss: 0.00047510533 Prediction: if you want you
'''