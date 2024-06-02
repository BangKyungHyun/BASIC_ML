# https://bioinformaticsandme.tistory.com/241
# TENSORFLOW 2.0에서 수행되는 기본적인 이미지 분류 신경망 학습 과정
# 0~9 사이의 10개 카테고리 이미지를 분류하기 위한 뉴럴 네트워크 모델
# 고수준 API인 케라스(tf.keras) 사용
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import os

print("pandas version: ", pd.__version__)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tensorflow version 확인
print('tensorflow version =',tf.__version__)

# 1. 숫자 mnist 데이터셋
# 이미지 분류의 재료가 되는 mnist 데이터셋
# 7만개의 흑백 이미지로 구성
# 각 이미지는 낮은 해상도로 개별 숫자를 나타냄
# 이미지 분류 뉴럴 네트워크 모델에 6만개 이미지 사용
# 만들어진 네트워크 이미지 분류 정확도를 평가하기 위해 나머지 1만개 이미지 사용
mnist = tf.keras.datasets.mnist

# load_data() 함수로 넘파이 배열을 반환
# 모든 이미지는 28*28 픽셀의 넘파이 배열(픽셀값은 0~255)
# Label은 0~9의 정수배열 (숫자 이미지의 클래스를 나타냄, 각 이미지는 1개의 Label에 매핑됨)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 점검
print("2. 데이터 점검")

# 2-1. 모델 훈련 전 train/test 데이터셋 값 및 형태(shape) 확인하기
print("2-1. 모델 훈련 전 train/test 데이터셋 값 및 형태(shape) 확인하기")

# 2-1-1. 모델 훈련 전 train/test 0번째 데이터 값 확인하기
print("\n2-1-1. 모델 훈련 전 train/test 0번째 데이터 값 확인하기")
print("train_images 0 번째 값 확인 :\n",train_images[0])

#lables_cnt = 60000
#for ii in range(lables_cnt):
#    print("train_labels ",ii,"번째 값 확인 :", train_labels[ii])

print("train_labels 0 번째 값 확인 :  ",train_labels[0])
print("train_labels       값 확인 :  ",train_labels)
print("test_images  0 번째 값 확인 :\n",test_images[0])
print("test_labels  0 번째 값 확인 :  ",test_labels[0])

# 2-1-2. 모델 훈련 전 train/test 데이터셋 형태(shape) 확인하기(reshape 이전 28*28 형태)
print("\n2-1-2 모델 훈련 전 train/test 데이터셋 형태(shape) 확인하기(reshape 이전 28*28 형태)")
print("reshape 이전 train_images.shape (6,0000개 이미지, 28*28 픽셀) => (60000, 28, 28) 표시 :",train_images.shape)
print("reshape 이전 train_labels.shape (train set의 각 lable은 0~9) => (60000,)        표시 :",train_labels.shape)
print("reshape 이전 test_images.shape  (1,0000개 이미지, 28*28 픽셀) => (60000, 28, 28) 표시 :",test_images.shape)
print("reshape 이전 test_labels.shape  (test set의 각 lable은 0~9)  => (10000,)        표시 :",test_labels.shape)

# 2-1-3. 모델 훈련 전 train/test 데이터셋 행의 갯수와 열의 갯수 확인하기(reshape 이전 28*28 형태)
# a = np.array( [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ])
# shape[0], shape[1]를 이용하여 전체 행의 갯수와 열의 갯수를 반환받을 수 있다.
# a.shape[0] 결과는 4임(shape 튜플의 첫번째 요소는 4)
# a.shape[1] 결과는 3임(shape 튜플의 두번째 요소는 3)
print("\n2-1-3. 모델 훈련 전 train/test 데이터셋 행의 갯수와 열의 갯수 확인하기(reshape 이전 28*28 형태)")
print("reshape 이전 train_images.shape[0] (6,0000개 이미지, 28*28 픽셀) => 60000 표시 :",train_images.shape[0])
print("reshape 이전 train_images.shape[1] (6,0000개 이미지, 28*28 픽셀) => 28    표시 :",train_images.shape[1])
print("reshape 이전 train_images.shape[2] (6,0000개 이미지, 28*28 픽셀) => 28    표시 :",train_images.shape[2])

# 모델 훈련 전 정확한 모델링을 위한 데이터셋 전 처리
# 픽셀 값의 범위가 0~255라는 것을 확인
#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# 28*28 2차원의 배열을 784 픽셀의 1차원 배열 형태로 만들기 위해 reshape 한다.
print("\n784 픽셀 형태로 만들기 위해 reshape 한다.")
train_images = train_images.reshape([-1, 784])
test_images  = test_images.reshape([-1, 784])

# 2-1-4. 모델 훈련 전 train/test 데이터셋 형태(shape) 확인하기(reshape 이후 786 형태)
print("\n2-1-4 모델 훈련 전 train/test 데이터셋 형태(shape) 확인하기(reshape 이후 786 형태)")
print("reshape 이후 train_images.shape (6,0000개 이미지, 786 픽셀    => (60000, 786) 표시 :",train_images.shape)
print("reshape 이후 train_labels.shape (train set의 각 lable은 0~9) => (60000,)     표시 :",train_labels.shape)
print("reshape 이후 test_images.shape  (1,0000개 이미지, 786 픽셀)   => (10000, 786) 표시 :",test_images.shape)
print("reshape 이후 test_labels.shape  (test set의 각 lable은 0~9)  => (10000,)     표시 :",test_labels.shape)

# 2-1-5. 모델 훈련 전 train/test 데이터셋 행의 갯수와 열의 갯수 확인하기(reshape 이후 786 형태)
print("\n2-1-5. 모델 훈련 전 train/test 데이터셋 행의 갯수와 열의 갯수 확인하기(reshape 이후 786 형태)")
print("reshape 이후 train_images.shape[0] (6,0000개 이미지,786 픽셀) => 60000 표시 :",train_images.shape[0])
print("reshape 이후 train_images.shape[1] (6,0000개 이미지,786 픽셀) =>   784 표시 :",train_images.shape[1])

# 원본 데이터는 각 픽셀이 0~255까지의 숫자로 음영의 강도를 표시함
# train, test set 모두 255로 나누어 범위값을 0~1사이로 표준화
train_images = train_images / 255.
test_images  = test_images  / 255.

#print("\ntrain_images[0] :\n",train_images[0])
#print("  test_images[0]  :\n",test_images[0])

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])

# train_labels 데이터를 one-hot으로 변환하여 10열로 y에 대입한다.
Y = tf.placeholder(tf.float32, [None, nb_classes])

# 가중치는 784개 행과 10개 열로 구성된다.
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# batch_xs, batch_ys = mnist.train.next_batch(100)
# Hypothesis (using softmax)
# 점수로 환산한 가설
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss (실제와 가설간의 차이 계산)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

# 예측한 결과를 argmax을 통해 0~9사이의 값으로 만든다
prediction = tf.argmax(hypothesis, 1)

# 예측한 결과와 Y 데이터를 비교한다.
# y 값은 argmax를 통해 one-hot에서 숫자로 바꾼다.
is_correct = tf.equal(prediction, tf.argmax(Y, 1))

# Calculate accuracy # 이것들을 평균낸다
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 1000000000
batch_size = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        # train_images.shape[0] (60,000) / batch_size로 나눈다.
        total_batch_cnt = int(train_images.shape[0] / batch_size)
        #print("epoch=", epoch, "train_images.shape[0] =", train_images.shape[0], "batch_size =", batch_size,
        #      "total_batch_cnt =", total_batch_cnt)

        for i in range(total_batch_cnt):
            s_idx = int(train_images.shape[0] * i / total_batch_cnt)
            e_idx = int(train_images.shape[0] * (i + 1) / total_batch_cnt)

            #print('s_idx : ', s_idx)
            #print('e_idx : ', e_idx)

            # total_batch_cnt (예 : 3만개) 갯수(2번)만큼 읽는다.
            # 0~30,000은 0부터 29,999까지 30,000개를 읽는다. slice 특성
            # 30,000~60,000은 30,000부터 59,999까지 30,000개를 읽는다. slice 특성
            batch_xs = train_images[s_idx: e_idx]
            batch_ys = train_labels[s_idx: e_idx]

            #print('batch_xs : ', batch_xs)
            #print('batch_xs.shape : ', batch_xs.shape)
            #print('batch_ys       : ', batch_ys)
            #아래는 batch_ys 실제 데이터
            #batch_ys: [3 9 3 7 5 3 2 6 4 7 5 8 9 0 6 3 2 5 4 3]
            # 아래는 batch_ys.shape 실제 데이터
            #print('batch_ys.shape : ', batch_ys.shape)
            #batch_ys.shape :  (20,)
            #Y_one_hot = tf.one_hot(batch_ys, nb_classes) #오류뱔생

            Y_one_hot = np.eye(nb_classes)[batch_ys]
            #print('Y_one_hot       :\n', Y_one_hot)
            # 아래는 Y_one_hot 실제 데이터
            # [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
            # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            # [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
            # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
            # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            # [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]
            #print('Y_one_hot.shape :'  , Y_one_hot.shape)
            # 아래는 Y_one_hot.shape 실제 데이터
            # Y_one_hot.shape : (20, 10)

            _, cost_val, w_val, y_val, h_val, p_val = sess.run([train, cost, W, Y, hypothesis, prediction],
                                                        feed_dict={X: batch_xs, Y: Y_one_hot})

            avg_cost += cost_val / total_batch_cnt

            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

            #print(nowDatetime,'Epoch:', epoch,'i', i,'cost :',cost_val, 'avg_cost :',avg_cost)
            #  "\nw_val[783]:\n",w_val[783])

            #print(nowDatetime, 'Epoch:', '%07d' % (epoch), 'i:',i, 'cost =',cost_val,
            #      'avg cost=','{:.9f}'.format(avg_cost), '\nweight[1]:\n', w_val[1],'\nhypothesis[1]:\n', h_val[1],
            #      '\nprediction[1]:\n', p_val[1], '\ny[1]=',y_val[1],'\ny[1]=',tf.argmax(y_val[1]))

        if epoch % 100000 == 0:
           now = datetime.datetime.now()
           nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
           print(nowDatetime,'Epoch:','%07d' % (epoch), 'avg cost=','{:.9f}'.format(avg_cost),'\nweight[0]:\n',w_val[0],
                             '\nhypothesis[0]:\n',h_val[0],'\nprediction[0]:\n',p_val[0])


    # test_labels 데이터를 10000*10 형태의 one-hot 를 구성한다.
    Y_one_hot = np.eye(nb_classes)[test_labels]

    #10,000 * 786 형태의 test_images와 786*10 형태의 가중치를 곱하여 argmax 한 값(0~9사의 값으로 표시)과
    #y를 one-hot 후 argmax 한 값이 같은 평균값을 계산
    #1. accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    #2. is_correct = tf.equal(prediction, tf.argmax(Y, 1))
    #3. prediction = tf.argmax(hypothesis, 1)
    #4. hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)  # W는 Train를 통해 이미 결정된 상태
    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess, feed_dict={X: test_images, Y: Y_one_hot}))

    ## 테스트 이미지 1개를 무작위로 가져옴
    r = random.randint(0, test_images.shape[0] - 1)
    ## 가져온 이미지 값을 출력함
    print('label : ', test_labels[r:r + 1]) # slice 특성성 r위치만 읽는다.
    ## 예측한 이미지 값을 출력함 - prediction을 위해서 hyothesis를 계산하기 위해 X값이 입력되어야 함
    print('Prediction : ', sess.run(tf.argmax(hypothesis, 1), feed_dict={X: test_images[r:r + 1]}))

    plt.imshow(
        test_images[r:r + 1].reshape(28, 28),
               cmap='Greys',
               interpolation='nearest'
    )
    plt.show()
# '''