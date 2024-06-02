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
# tf.contrib.layers.fully_connected 의 contrib 문제를 해결하기 위한
# _layers.fully_connected를 사용하기 위해  tf_slim.layers 를 추가함 (2022.9.14)

'''
코드의 전반적인 내용을 보면 학습시키고자 하는 문자열은 'hihello' 이다.
idx2char은 각 알파벳에 인덱스를 부여하는데, 그 인덱스를 주면 그에 해당하는 알파벳을 반환해 주도록 만든 것이다.
그런 의미로 xdata를 해석해보면 'hihell'이 된다.

x_one_hot은 x_data 를 one hot encoding한 것인데 역시 차례대로 h,i,h,e,l,l을 의미하고 있다.
y_data는 예측하려는 문자열로 'ihello'를 나타낸다.
'''

# 어휘, 원핫 인코딩 벡터 입력과 출력 라벨
idx2char = ['h', 'i', 'e', 'l', 'o'] # 어휘 리스트, 인덱스(h=0, i=1, e=2, l=3, o=4)
print ('idx2char = ', idx2char)
'''
idx2char =  ['h', 'i', 'e', 'l', 'o']
'''

# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]    # 1 x 6 구조의 출력 라벨, ihello

# 모델 파라미터 설정
input_dim       = 5     # RNN cell의 입력 크기, 어휘의 갯수  # one-hot size
hidden_size     = 5     # RNN cell의 출력 (상태) 크기
batch_size      = 1     # 입력되는 배치의 크기, 문자열 시퀸스 갯수
sequence_length = 6     # RNN의 타임 스템(윈도우) 크기, 입력되는 문자열의 크기  |ihello| == 6
learning_rate   = 0.1   # 학습 속도
num_classes     = 5     # 최종 출력 클래스 크기

# X에는 x_data가 들어갈 placeholder이고 차례대로 [batch_size, sequence_length, input_dim] 이다.
# Y에는 y_data가 들어갈 placeholder이고 차례대로 [batch_size, sequence_length] 이다.
# 여기서는 batch_size가 1 이지만 None로 준 것은 사실 batch를 몇개를 받아와도 상관 없음
# sequence_length는 one hot의 개수로 6이 되고 input_dim은 x 벡터의 차원으로 5가 된다.

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

'''
RNN 생성
#cell = tf.contrib.rnn.BasicLSTMCell(
## contrib 아래와 같이 수정함 (2022.9.14)
# BasicRNNCell 객체를 생성하여 RNN의 셀을 구성
# hidden_size = 5
'''
cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True) # 초기 상태 지정
print("cell= ", cell)
'''
cell=  <keras.layers.rnn.legacy_cells.BasicLSTMCell object at 0x000001CE9BAFC580>
'''
initial_state = cell.zero_state(batch_size, tf.float32)
'''

#dynamic_rnn() 함수에 생성된 셀을 인수로 전달하여 RNN 구성
tf.nn.dynamic_rnn(cell, inputs, initial_state=None, dtype=None) 함수
• 지정된 셀을 사용하여 RNN 을 생성 ( RNN의 출력과 셀의 상태를 리턴 )
• cell: RNN 셀 인스턴스
• inputs: RNN 입력 (3-D 구조의 텐서, [batch_size, sequence_length, input_dim])
• intial_state: RNN의 초기 상태
• dtype: 초기 상태와 출력의 자료형
'''
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
print("outputs 1 = ", outputs)
print("_states= ", _states)
'''
outputs=  Tensor("rnn/transpose_1:0", shape=(1, 6, 5), dtype=float32)
_states=  LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(1, 5) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(1, 5) dtype=float32>)
'''

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])  # 3-D 구조를 2-D 구조로 변환, hidden_size=5
print("X_for_fc= ", X_for_fc)
'''
X_for_fc=  Tensor("Reshape:0", shape=(6, 5), dtype=float32)
'''
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
'''
완전 연결 계층
 outputs = tf.contrib.layers.fully_connected(
# contrib 아래와 같이 수정함 (2022.9.14)
 은닉 상태의 출력을 완전 연결 계층의 입력으로 연결
• 활성화 함수는 적용하지 않음
• 선형 함수 그대로 사용

 tf.contrib.layers.fully_connected(inputs, num_outputs,activation_fn) 함수 사용
• 인수
• inputs: 입력, 2-D 텐서, [batch_size, hidden_size]
• num_outputs: 출력의 갯 수
• activation_fn: 활성화 함수
– 디폴트는 ReLU
– None 이면 활성화 함수는 적용하지 않음
• [sequenth_lengh, num_classes] 구조의 출력을 리턴
'''
outputs2 = _layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
print("outputs 2 = ", outputs2)
#outputs 2 =  Tensor("fully_connected/BiasAdd:0", shape=(6, 5), dtype=float32)
# reshape out for sequence_loss
outputs3 = tf.reshape(outputs2, [batch_size, sequence_length, num_classes])
print("outputs 3 = ", outputs3)
#outputs 3 =  Tensor("Reshape_1:0", shape=(1, 6, 5), dtype=float32)

weights = tf.ones([batch_size, sequence_length])
print("weights = ", weights)
#weights =  Tensor("ones:0", shape=(1, 6), dtype=float32)

#sequence_loss = tf.contrib.seq2seq.sequence_loss(
# contrib 아래와 같이 수정함 (2022.9.14)

'''
시퀀스 모델 훈련
 시퀀스에 대해 교차 엔트로피 손실 함수 적용
• tf.contrib.seq2seq.sequence_loss(logits, targets, weights) 함수
사용
• logits: 모델의 출력, [batch_size, sequence_length, num_classes]
• targets: 모델의 출력 라벨, [batch_size, sequnce_lengh]
– 출력 중 라벨에 해당하는 인덱스
• weights: 가중치 마스크
– 시퀀스 (타임 스텝) 중에서 예측에 사용되는 가중치를 선택
– [batch_size, sequence_length]
– 모두 사용하려면 1로 값을 설정
• 디폴트로 시퀀스의 평균 손실 스칼라 (0-D 구조) 값 리턴
'''
sequence_loss = tfa.seq2seq.sequence_loss(logits=outputs3, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

'''
with 구문 세션 생성
 파이썬 with 구문을 사용한 세션 생성
• 파이썬의 컨텍스트 매니저(context manager)
• with 구문을 사용하여 리소스(파일, 세션 등)의 사용을 특정 블럭 내에서
동작하도록 제한
• 블록을 끝에서 자동으로 리소스를 해제
• with ~ as ~구문으로 세션을 생성하고 블록의 끝에서 자동으로
세션을 해제하는 close()를 호출
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _, result, weights_val, outputs_val, outputs2_val, outputs3_val, X_for_fc_val = \
            sess.run([loss, train, prediction, weights, outputs, outputs2, outputs3, X_for_fc ],
                                                          feed_dict={X: x_one_hot, Y: y_data})
        #result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "Y data : ", y_data, '\nweights =\n',weights_val,
               '\noutputs =\n', outputs_val, '\nX_for_fc  =\n',X_for_fc_val, '\noutputs 2 =\n', outputs2_val,
               '\noutputs 3 =\n', outputs3_val)

        # print char using dic
        print("\tnp.squeeze(result): ", np.squeeze(result))
        # squeeze () : 함수로 2-D구조를 1-D구조로 변환, 크기가 1 인 axis 제거
        #
        result_str = [idx2char[c] for c in np.squeeze(result)]
        # join 메서드로 리스트를 문자열로 변환
        print("\tidx2char: ", idx2char)

        print("\tPrediction str: ", ''.join(result_str))
'''
0 loss: 1.71584 prediction:  [[2 2 2 3 3 2]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  eeelle
1 loss: 1.56447 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
2 loss: 1.46284 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
3 loss: 1.38073 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
4 loss: 1.30603 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
5 loss: 1.21498 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
6 loss: 1.1029 prediction:  [[3 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  lhlllo
7 loss: 0.982386 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihlllo
8 loss: 0.871259 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihlllo
9 loss: 0.774338 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
10 loss: 0.676005 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
...
'''