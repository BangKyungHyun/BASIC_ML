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

#pprint 모듈은 임의의 파이썬 데이터 구조를 인터프리터의 입력으로 사용할 수 있는 형태로 “예쁘게 인쇄”할 수 있는 기능을 제공합니다.
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

print ('\n========== 1. cell ==============')

with tf.variable_scope('one_cell') as scope:
    # One cell RNN input_dim (4) -> output_dim (2)
    hidden_size = 2

    #BasicRNNCell 객체를 생성하여 RNN의 셀을 구성

    '''
     tf.nn.rnn_cell.BasicRNNCell 클래스
    • 기본적인 RNN 셀의 클래스
    • 생성자
    __init__(
        num_units,
        activation=None,
        ....
    )
    • num_units: 셀의 유닛(상태) 수
    • activation: 활성화 함수, 디폴트로 tanh
    '''
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    print('cell.output_size =',cell.output_size, 'cell.state_size =', cell.state_size)
    #cell.output_size = 2 cell.state_size = 2

# x_data로 h = [1,0,0,0] 을 대입하면 input dimension은 4가 된다.
# hidden_size가 2이므로 output dimension은 2가 된다.
# 입력으로 h 만을 사용
    x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]]
    pp.pprint(x_data)
    # array([[[1., 0., 0., 0.]]], dtype=float32)

    # 우리가 만든 cell과 입력 값인 x_data를 인자로 넘겨주고
    # 해당 스텝의 output과 다음 스테이트로 넘어가는 출력 _state를 만든다

    #dynamic_rnn() 함수에 생성된 셀을 인수로 전달하여 RNN 구성
    '''
     tf.nn.dynamic_rnn(cell, inputs, initial_state=None, dtype=None) 함수
    • 지정된 셀을 사용하여 RNN 을 생성
    • RNN의 출력과 셀의 상태를 리턴
    • cell: RNN 셀 인스턴스
    • inputs: RNN 입력
    • 3-D 구조의 텐서, [batch_size, sequence_length, input_dim]
    • intial_state: RNN의 초기 상태
    • dtype: 초기 상태와 출력의 자료형
    '''
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

    # array([[[-0.47344998,  0.70767045]]], dtype=float32)

#hidden size을 2로 설정하여 RNN학습을 시킨 결과를 나타낸다. 입력으로는 [1,1,4]의 형태로 입력되나
#hidden size의 설정으로  Output 형태는 (1,1,2)의 형태가 된다.

print ('\n========== 2. sequances ==============')

with tf.variable_scope('two_sequances') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence_length: 5 (cell 갯수)
    hidden_size = 2
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
# x_data에  h, e, l, l, o 의 5개 철자를 입력하면 x_data.shape = (1, 5, 4) 이 된다.
# 즉 sequence(cell) length 가 5가 된다.
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    print('x_data.shape =',x_data.shape)
    # x_data.shape = (1, 5, 4)

    pp.pprint(x_data)
#   array([[[1., 0., 0., 0.],
#           [0., 1., 0., 0.],
#           [0., 0., 1., 0.],
#           [0., 0., 1., 0.],
#           [0., 0., 0., 1.]]], dtype=float32)

# output data shape = [1,5,2]  5: sequence length,  2 : hidden_size
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
''' 
  aray([[[ 0.19479147, -0.6857029 ],
         [ 0.4439441 , -0.8175473 ],
         [-0.78480214, -0.8009785 ],
         [-0.94387966, -0.10135241],
         [ 0.31120533,  0.86386037]]], dtype=float32)
'''
# 위 예제는 sequence에 대해 알려준다. RNN의 특징은 이전의 상태에 따라 결과가 달라진다는 점이다.
# 그래서 sequence에 대한 설정이 필요한데 위의 예제는 sequence를 5로 설정하여 cell이 5개가 나타나는 걸
# 보여준다. 이로 인해 input data형태도 (1,5,4)로 output 형태도 (1,5,2)로 바뀐 것 알 수 있다.

print ('\n========== 3. batches ==============')

with tf.variable_scope('3_batches') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch size : 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    print('x_data.shape =',x_data.shape)
    # x_data.shape = (3, 5, 4)  -> 4 인 이유 :  l은 내부적으로 [0, 0, 1, 0] 이므로
    pp.pprint(x_data)
#  array([[[1., 0., 0., 0.],
#          [0., 1., 0., 0.],
#          [0., 0., 1., 0.],
#          [0., 0., 1., 0.],
#          [0., 0., 0., 1.]],
#
#         [[0., 1., 0., 0.],
#          [0., 0., 0., 1.],
#          [0., 0., 1., 0.],
#          [0., 0., 1., 0.],
#          [0., 0., 1., 0.]],
#
#         [[0., 0., 1., 0.],
#          [0., 0., 1., 0.],
#          [0., 1., 0., 0.],
#          [0., 1., 0., 0.],
#          [0., 0., 1., 0.]]], dtype=float32)

    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

# output data shape = [3,5,2]  3 : batch size 5: sequence length,  2 : hidden_size
# array([[[ 3.46301459e-02, -1.37192924e-02],
#         [ 3.10476851e-02, -7.06022084e-02],
#         [-8.51348341e-02, -1.41251355e-01],
#         [-1.79029644e-01, -1.96395680e-01],
#         [-3.23956497e-02, -1.37332782e-01]],
#
#        [[ 6.10939134e-03, -6.34710938e-02],
#         [ 1.34086922e-01, -4.65336598e-05],
#         [-2.15673391e-02, -8.40736851e-02],
#         [-1.30295336e-01, -1.53881654e-01],
#         [-2.09408388e-01, -2.06000865e-01]],
#
#        [[-1.04836628e-01, -8.75200927e-02],
#         [-1.84226707e-01, -1.57347068e-01],
#         [-1.16362594e-01, -1.85151219e-01],
#         [-1.04003198e-01, -2.01805443e-01],
#         [-2.23575681e-01, -2.37386659e-01]]], dtype=float32)

print ('\n========== 4. 3_batches_dynamic_length ==============')

with tf.variable_scope('3_batches_dynamic_length') as scope:
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch size : 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

#    array([[[1., 0., 0., 0.],
#            [0., 1., 0., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 0., 1.]],
#
#           [[0., 1., 0., 0.],
#            [0., 0., 0., 1.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.]],
#
#           [[0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 1., 0., 0.],
#            [0., 1., 0., 0.],
#            [0., 0., 1., 0.]]], dtype=float32)
#
    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)
# 위 예제와 비교시 sequence_length=[5, 3, 4] 가 추가됨
#    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,  sequence_length=[5, 3, 4], dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

#array([[[ 1.3476002e-01,  7.6806903e-02],
#        [ 7.5070664e-02,  1.1127783e-01],
#        [ 2.4519481e-02,  1.5448720e-02],
#        [-2.2712111e-02, -5.4266851e-02],
#        [ 5.4194305e-02, -1.3745511e-01]],
#
#       [[-1.1308891e-02,  6.3265264e-02],
#        [ 7.1955755e-02, -5.2409295e-02],
#        [-1.0839518e-04, -8.4491462e-02],
#        [ 0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00]],
#
#       [[-4.3765735e-02, -6.3172370e-02],
#        [-8.4722728e-02, -9.5484063e-02],
#        [-8.6707406e-02,  8.4551219e-03],
#        [-7.4871317e-02,  6.3292645e-02],
#        [ 0.0000000e+00,  0.0000000e+00]]], dtype=float32)

print ('\n========== 5. initial_state ==============')

with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

#    array([[[1., 0., 0., 0.],
#            [0., 1., 0., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 0., 1.]],
#
#           [[0., 1., 0., 0.],
#            [0., 0., 0., 1.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 1., 0.]],
#
#           [[0., 0., 1., 0.],
#            [0., 0., 1., 0.],
#            [0., 1., 0., 0.],
#            [0., 1., 0., 0.],
#            [0., 0., 1., 0.]]], dtype=float32)
#
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,initial_state=initial_state,dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
'''    
array([[[ 0.07701926, -0.11205729],
        [ 0.02315595, -0.00213687],
        [-0.0921004 , -0.07425977],
        [-0.16265993, -0.14809075],
        [-0.08534513, -0.03630036]],

       [[-0.05715192,  0.0793068 ],
        [ 0.0034889 ,  0.17200296],
        [-0.11203442,  0.04121402],
        [-0.18250293, -0.05364097],
        [-0.22334151, -0.14234784]],

       [[-0.1029453 , -0.07560786],
        [-0.16969241, -0.1504054 ],
        [-0.2404289 , -0.08950305],
        [-0.22570407, -0.01798118],
        [-0.20190182, -0.11563779]]], dtype=float32)
'''

print ('\n========== 6. Create input data ==============')
# Create input data
batch_size=3
sequence_length=5
input_dim=3

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)  # batch, sequence_length, input_dim

'''
array([[[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.],
        [12., 13., 14.]],

       [[15., 16., 17.],
        [18., 19., 20.],
        [21., 22., 23.],
        [24., 25., 26.],
        [27., 28., 29.]],

       [[30., 31., 32.],
        [33., 34., 35.],
        [36., 37., 38.],
        [39., 40., 41.],
        [42., 43., 44.]]], dtype=float32)

'''

print ('\n========== 7. generated_data ==============')

with tf.variable_scope('generated_data') as scope:
    # One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

'''
array([[[ 1.52272359e-02,  5.30151799e-02, -6.54325038e-02,       -1.53822601e-01, -6.68164864e-02],
       [ 6.95032775e-02,  1.30128460e-02, -2.29598749e-02,        -3.44258815e-01, -2.64930248e-01],
       [ 4.29045819e-02,  6.36609038e-04,  6.99947178e-02,        -3.91795456e-01, -3.45450938e-01],
       [ 1.67215392e-02,  2.22673061e-05,  1.40800148e-01,        -3.73503238e-01, -3.63316029e-01],
       [ 5.88076608e-03,  1.02042213e-06,  1.47978380e-01,        -3.39117378e-01, -3.64247352e-01]],

      [[ 1.69384293e-03,  4.46945201e-08,  5.43553531e-02,        -1.97703719e-01, -3.77927045e-03],
       [ 6.07678725e-04,  2.03376649e-09,  6.32949024e-02,        -2.28809863e-01, -5.09004248e-03],
       [ 2.07362231e-04,  9.36671851e-11,  4.86890748e-02,        -2.08860621e-01, -5.52336639e-03],
       [ 6.95093986e-05,  4.31208931e-12,  3.25732492e-02,        -1.82573557e-01, -5.66045195e-03],
       [ 2.32599796e-05,  1.97544916e-13,  2.08154377e-02,        -1.58099279e-01, -5.70127089e-03]],

      [[ 7.52317510e-06,  9.12109726e-15,  7.26234261e-03,        -1.01032063e-01, -1.78191040e-05],
       [ 2.59628791e-06,  4.09495961e-16,  7.11687841e-03,        -1.12364352e-01, -2.39990386e-05],
       [ 8.79667994e-07,  1.84880177e-17,  4.95828269e-03,        -1.00193374e-01, -2.61111127e-05],
       [ 2.96607254e-07,  8.34868584e-19,  3.17313452e-03,        -8.63737166e-02, -2.68291751e-05],
       [ 9.99954892e-08,  3.76508575e-20,  1.99189573e-03,        -7.40062371e-02, -2.70726032e-05]]], dtype=float32)

'''

print ('\n========== 8. MultiRNNCell ==============')
with tf.variable_scope('MultiRNNCell') as scope:
    # Make rnn
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
#원본    cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 3 layers
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell] , state_is_tuple=True) # 3 layers

    # rnn in/out
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    print("dynamic rnn 1 : ", outputs)

# dynamic rnn 1 :  Tensor("MultiRNNCell/rnn/transpose_1:0", shape=(3, 5, 5), dtype=float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

'''
array([[[-0.1904979 ,  0.03690426, -0.17290713,  0.17309634,
         -0.12360378],
        [-0.31040913, -0.07388689, -0.41190505,  0.5187418 ,
         -0.27695253],
        [-0.3409478 , -0.28203252, -0.56108063,  0.75895137,
         -0.4287445 ],
        [-0.32927474, -0.42616564, -0.5901625 ,  0.8524164 ,
         -0.55928177],
        [-0.29729787, -0.5111115 , -0.5192467 ,  0.87894887,
         -0.661798  ]],

       [[-0.14509739, -0.52126694, -0.2226255 ,  0.68361616,
         -0.30466694],
        [-0.19162688, -0.56870306, -0.23875505,  0.80000347,
         -0.5573189 ],
        [-0.17601524, -0.6028492 , -0.20613946,  0.8282869 ,
         -0.7208195 ],
        [-0.15174292, -0.62279075, -0.16602147,  0.8307698 ,
         -0.8222492 ],
        [-0.12823346, -0.63808805, -0.13050757,  0.82423884,
         -0.88288254]],

       [[-0.06191263, -0.6445084 , -0.07617809,  0.7367203 ,
         -0.44540304],
        [-0.08115463, -0.66021097, -0.06442347,  0.7869489 ,
         -0.72015756],
        [-0.07190248, -0.6732937 , -0.05496358,  0.79154855,
         -0.8606422 ],
        [-0.06065909, -0.6834606 , -0.04436179,  0.7881786 ,
         -0.9280936 ],
        [-0.05045541, -0.6924695 , -0.03476598,  0.7834923 ,
         -0.9596525 ]]], dtype=float32)
'''
print ('\n========== 9. dynamic_rnn ==============')
with tf.variable_scope('dynamic_rnn') as scope:
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32,sequence_length=[1, 3, 2])
    # lentgh 1 for batch 1, lentgh 2 for batch 2

    print("dynamic rnn 2 : ", outputs)
#dynamic rnn 2 :  Tensor("dynamic_rnn/rnn/transpose_1:0", shape=(3, 5, 5), dtype=float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

'''    
array([[[ 1.51539057e-01,  6.82343743e-05, -6.00755736e-02,
         -5.01740836e-02,  3.05741392e-02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]],

       [[ 2.43308663e-04,  5.12871985e-08,  6.35912046e-02,
         -1.75319066e-08,  1.33458991e-04],
        [ 4.95463100e-05,  2.50052690e-09,  9.93718058e-02,
         -2.53430898e-09,  3.16551304e-05],
        [ 1.00331117e-05,  1.21382238e-10,  1.19042665e-01,
         -2.89404251e-10,  5.64991615e-06],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]],

       [[ 8.04064086e-08,  1.39045608e-14,  4.76133358e-03,
         -2.37939919e-15,  3.65193138e-08],
        [ 1.61914837e-08,  6.72028264e-16,  7.51628727e-03,
         -3.50178270e-16,  7.14421411e-09],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]]], dtype=float32)

'''
print ('\n========== 10. bi-directional ==============')
with tf.variable_scope('bi-directional') as scope:
    # bi-directional rnn
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data,
                                                      sequence_length=[2, 3, 1],
                                                      dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    print('sess.run(outputs) =\n')
    pp.pprint(sess.run(outputs))
    '''
    (   array([[[ 1.09408416e-01,  4.97364020e-03, -1.61741659e-01,
         -4.95257825e-02,  1.22778624e-01],
        [ 1.58639148e-01,  1.63448632e-01, -1.95927501e-01,
          7.86198229e-02, -4.88407398e-03],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]],

       [[ 4.72705392e-03,  1.15301097e-02, -9.11348543e-05,
          1.83643351e-04, -6.88394010e-01],
        [ 3.15734092e-03,  6.70887483e-03, -1.05452207e-04,
          2.67465912e-05, -8.95874739e-01],
        [ 1.49348739e-03,  3.11685703e-03, -1.01702019e-04,
          4.17045612e-06, -9.43637192e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]],

       [[ 9.00113373e-05,  2.03298303e-04, -2.49205439e-08,
          2.33771136e-08, -7.48544812e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00]]], dtype=float32),
    array([[[ 1.8131739e-02, -5.6976773e-02, -9.5710285e-02,  1.4719754e-01,
          1.4199676e-01],
        [-1.3576356e-01, -1.4950953e-02, -3.2694615e-02,  6.2403072e-02,
          2.3024486e-01],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00]],

       [[-8.3081193e-02, -3.6834419e-05, -4.7028734e-06,  1.4083780e-03,
          2.0585811e-01],
        [-5.8014221e-02, -5.7728298e-06, -4.8215702e-07,  4.7893022e-04,
          1.7716576e-01],
        [-4.0795341e-02, -7.6060019e-07, -5.1612307e-08,  1.5581747e-04,
          1.4691217e-01],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00]],

       [[-1.2910591e-02, -3.1094334e-09, -5.1592106e-11,  6.0801808e-06,
          8.5062273e-02],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00]]], dtype=float32))
    '''
    print('sess.run(states) =\n')
    pp.pprint(sess.run(states))
    '''
    (   LSTMStateTuple(c=array([[ 3.7522233e-01,  6.5011853e-01, -2.0515029e-01,  2.3378399e-01,
        -6.8122298e-03],
       [ 3.1012320e-03,  2.9833660e+00, -1.0170203e-04,  1.1893079e-03,
        -2.9699862e+00],
       [ 1.9031858e-04,  9.9985087e-01, -2.4920544e-08,  8.6605447e-05,
        -9.9993002e-01]], dtype=float32), h=array([[ 1.5863915e-01,  1.6344863e-01, -1.9592750e-01,  7.8619823e-02,
        -4.8840740e-03],
       [ 1.4934874e-03,  3.1168570e-03, -1.0170202e-04,  4.1704561e-06,
        -9.4363719e-01],
       [ 9.0011337e-05,  2.0329830e-04, -2.4920544e-08,  2.3377114e-08,
        -7.4854481e-01]], dtype=float32)),
    LSTMStateTuple(c=array([[ 3.77018750e-02, -2.14417905e-01, -2.60900319e-01,
         3.17253590e-01,  2.37262473e-01],
       [-8.33319202e-02, -1.40537396e-01, -1.46162085e-04,
         1.69551396e-03,  3.26163888e-01],
       [-1.29113169e-02, -1.72349345e-02, -2.88887474e-08,
         6.32854926e-06,  1.22581996e-01]], dtype=float32), h=array([[ 1.8131739e-02, -5.6976773e-02, -9.5710285e-02,  1.4719754e-01,
         1.4199676e-01],
       [-8.3081193e-02, -3.6834419e-05, -4.7028734e-06,  1.4083780e-03,
         2.0585811e-01],
       [-1.2910591e-02, -3.1094334e-09, -5.1592106e-11,  6.0801808e-06,
         8.5062273e-02]], dtype=float32)))
    '''
print ('\n========== 11. flattern based softmax ==============')
# flattern based softmax
hidden_size=3
sequence_length=5
batch_size=3
num_classes=5

print('x_data 1 =\n')
pp.pprint(x_data) # hidden_size=3, sequence_length=4, batch_size=2
'''
array([[[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.],
        [12., 13., 14.]],

       [[15., 16., 17.],
        [18., 19., 20.],
        [21., 22., 23.],
        [24., 25., 26.],
        [27., 28., 29.]],

       [[30., 31., 32.],
        [33., 34., 35.],
        [36., 37., 38.],
        [39., 40., 41.],
        [42., 43., 44.]]], dtype=float32)
'''

x_data = x_data.reshape(-1, hidden_size)
print('x_data 2 =\n')
pp.pprint(x_data)
'''
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.],
       [ 9., 10., 11.],
       [12., 13., 14.],
       [15., 16., 17.],
       [18., 19., 20.],
       [21., 22., 23.],
       [24., 25., 26.],
       [27., 28., 29.],
       [30., 31., 32.],
       [33., 34., 35.],
       [36., 37., 38.],
       [39., 40., 41.],
       [42., 43., 44.]], dtype=float32)
'''

softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data, softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
print('outputs =\n')
pp.pprint(outputs)

'''
array([[[  25.,   28.,   31.,   34.,   37.],
        [  70.,   82.,   94.,  106.,  118.],
        [ 115.,  136.,  157.,  178.,  199.],
        [ 160.,  190.,  220.,  250.,  280.],
        [ 205.,  244.,  283.,  322.,  361.]],

       [[ 250.,  298.,  346.,  394.,  442.],
        [ 295.,  352.,  409.,  466.,  523.],
        [ 340.,  406.,  472.,  538.,  604.],
        [ 385.,  460.,  535.,  610.,  685.],
        [ 430.,  514.,  598.,  682.,  766.]],

       [[ 475.,  568.,  661.,  754.,  847.],
        [ 520.,  622.,  724.,  826.,  928.],
        [ 565.,  676.,  787.,  898., 1009.],
        [ 610.,  730.,  850.,  970., 1090.],
        [ 655.,  784.,  913., 1042., 1171.]]], dtype=float32)
'''

print ('\n========== sequence_loss ==============')
# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])
print("y_data : ", y_data)
#y_data :  Tensor("Const:0", shape=(1, 3), dtype=int32)

# [batch_size, sequence_length, emb_dim ]
#prediction = tf.constant([[[0.9], [0.9], [0.9]]], dtype=tf.float32)
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
#prediction = tf.constant([[[0.2, 0.7, 0.1, 0.7], [0.6, 0.2, 0.1, 0.7], [0.2, 0.9, 0.1, 0.7]]], dtype=tf.float32)
#prediction = tf.constant([[[0.2], [0.6], [0.2]]], dtype=tf.float32)

print("prediction: ", prediction)
# prediction:  Tensor("Const_1:0", shape=(1, 3, 2), dtype=float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

'''
 시퀀스에 대해 교차 엔트로피 손실 함수 적용
• tf.contrib.seq2seq.sequence_loss(logits, targets, weights) 함수 사용
• logits: 모델의 출력, [batch_size, sequence_length, num_classes]
• targets: 모델의 출력 라벨, [batch_size, sequnce_lengh]
  – 출력 중 라벨에 해당하는 인덱스
• weights: 가중치 마스크
  – 시퀀스 (타임 스텝) 중에서 예측에 사용되는 가중치를 선택
  – [batch_size, sequence_length]
  – 모두 사용하려면 1로 값을 설정
• 디폴트로 시퀀스의 평균 손실 스칼라 (0-D 구조) 값 리턴
'''
p_prediction = prediction
sequence_loss = tfa.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
print("sequence_loss: ", sequence_loss)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())
print("p_prediction : ", p_prediction.eval())

'''
Loss:  0.5967595
'''

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

prediction3 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
prediction4 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)
prediction5 = tf.constant([[[0, 1], [0, 1], [0, 1]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tfa.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tfa.seq2seq.sequence_loss(prediction2, y_data, weights)
sequence_loss3 = tfa.seq2seq.sequence_loss(prediction3, y_data, weights)
sequence_loss4 = tfa.seq2seq.sequence_loss(prediction4, y_data, weights)
sequence_loss5 = tfa.seq2seq.sequence_loss(prediction5, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval(),
      "Loss4: ", sequence_loss4.eval(),
      "Loss5: ", sequence_loss5.eval()
    )

'''
Loss1:  0.5130153 Loss2:  0.3711007 Loss3:  1.3132616 Loss4:  0.64659494 Loss5:  0.31326166
'''