import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
tf.compat.v1.disable_eager_execution()
import datetime


# read_data_sets() 를 통해 데이터를 객체형태로 받아오고
# 정답(label)은 한 자리 숫자로 저장되어 있기 때문에
# one_hot 옵션을 통해 정답(label) 을 one-hot 인코딩된 10자리 형태로 받아옴
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 28*28 2차원의 배열을 784 픽셀의 1차원 배열 형태로 만들기 위해 reshape 한다.
train_images = train_images.reshape([-1, 784])
test_images = test_images.reshape([-1, 784])

# 원본 데이터는 각 픽셀이 0~255까지의 숫자로 음영의 강도를 표시함
# train, test set 모두 255로 나누어 범위값을 0~1사이로 표준화
train_images = train_images / 255.
test_images = test_images  / 255.


# 입력노드, 은닉노드, 출력노드, 학습율, 반복횟수, 배치 개수 등 설정
learning_rate = 0.1  # 학습율
num_epochs = 10 # 반복횟수
batch_size = 20      # 한번에 입력으로 주어지는 MNIST 개수

input_nodes = 784     # 입력노드 개수
hidden_nodes = 100    # 은닉노드 개수
output_nodes = 10     # 출력노드 개수
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, input_nodes])
# train_labels 데이터를 one-hot으로 변환하여 10열로 y에 대입한다.
Y = tf.placeholder(tf.float32, [None, nb_classes])

# tf.nn.conv2d(input, filter, strides, padding, ....)
# 입력층의 출력 값. 컨볼루션 연산을 위해 reshape 시킴
A1 = tf.reshape(X, [-1, 28, 28, 1])   # image 28 X 28 X 1 (black/white)

# 1번째 컨볼루션 층
# 3X3 크기를 가지는 32개의 필터를 적용
# filter : 컨볼루션연산에 적용될 필터 [filter_height, filter_width, in_channels, out_channels]
# 예를 들어, 필터 크기 3 x 3, 입력채널 1개, 적용되는 필터 개수 32개일 경우 filter는 [3, 3, 1, 32]로 나타냄
F2 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
b2 = tf.Variable(tf.constant(0.1, shape=[32]))
# strides : 컨볼루션 연산을 위해 필터를 이동시키는 간격을 나타냄. 예를 들어 [1, 1, 1, 1]은 컨볼루션 연산을 위해 1칸씩 이동함
C2 = tf.nn.conv2d(A1, F2, strides=[1, 1, 1, 1], padding='SAME')
# 1번째 컨볼루션 연산을 통해 28 X 28 X 1  => 28 X 28 X 32

# relu
Z2 = tf.nn.relu(C2+b2)

# tf.nn.max_pool(value, ksize, strides, padding, ...)
# value : relu를 통과한 출력결과를 나타내고 pooling의 입력데이터로 들어옴
# ksize : ksize는 [1, height, width, 1] 형태로 표시함. [1, 2, 2, 1] 이라면 (2 x 2) 데이터중 가장 큰 값 1개를 찾아서 반환하는 의미
# strides : strides가 [1, 2, 2, 1] 일경우 max pooling 적용을 위하 2칸씩 이동하는 것을 의미
# padding : max pooling을 수행하기에는 데이터가 부족한 경우에 주변에 0 등으로 채워주는 역할을 함
# 1번째 max pooling을 통해 28 X 28 X 32  => 14 X 14 X 32
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 2번째 컨볼루션 층
F3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
b3 = tf.Variable(tf.constant(0.1, shape=[64]))

# 2번째 컨볼루션 연산을 통해 14 X 14 X 32 => 14 X 14 X 64
C3 = tf.nn.conv2d(A2, F3, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z3 = tf.nn.relu(C3+b3)

# 2번째 max pooling을 통해 14 X 14 X 64 => 7 X 7 X 64
A3 = P3 = tf.nn.max_pool(Z3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3번째 컨볼루션 층
F4 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
b4 = tf.Variable(tf.constant(0.1, shape=[128]))

# 3번째 컨볼루션 연산을 통해 7 X 7 X 64 => 7 X 7 X 128
C4 = tf.nn.conv2d(A3, F4, strides=[1, 1, 1, 1], padding='SAME')

# relu
Z4 = tf.nn.relu(C4+b4)

# 3번째 max pooling을 통해 7 X 7 X 128 => 4 X 4 X 128
A4 = P4 = tf.nn.max_pool(Z4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 4X4 크기를 가진 128개의 activation map을 flatten 시킴
A4_flat = P4_flat = tf.reshape(A4, [-1, 128*4*4])

# 출력층
W5 = tf.Variable(tf.random_normal([128*4*4, 10], stddev=0.01))
b5 = tf.Variable(tf.random_normal([10]))

# 출력층 선형회귀  값 Z5, 즉 softmax 에 들어가는 입력 값
Z5 = logits = tf.matmul(A4_flat, W5) + b5    # 선형회귀 값 Z5
A5 = tf.nn.softmax(Z5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=A5) )

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교함
predicted_val = tf.equal( tf.argmax(A5, 1), tf.argmax(Y, 1) )

# batch_size X 10 의 True, False 를 1 또는 0 으로 변환
accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 변수 노드(tf.Variable) 초기화

    for epoch in range(num_epochs):  # 100 번 반복수행
        avg_cost = 0

        total_batch_cnt = int(train_images.shape[0] / batch_size)

        for i in range(total_batch_cnt):
            s_idx = int(train_images.shape[0] * i / total_batch_cnt)
            e_idx = int(train_images.shape[0] * (i + 1) / total_batch_cnt)

            batch_xs = train_images[s_idx: e_idx]
            batch_ys = train_labels[s_idx: e_idx]

            Y_one_hot = np.eye(nb_classes)[batch_ys]

            _, cost_val, y_val, h_val \
                    = sess.run([train, cost, Y, logits],
                               feed_dict={X: batch_xs, Y: Y_one_hot,
                                          keep_prob: 0.7})

            avg_cost += cost_val / total_batch_cnt

        if epoch % 10 == 0:
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            print(nowDatetime, 'Epoch:', '%07d' % (epoch), 'avg cost=',
                      avg_cost)

    print('Learning Finished!')

    print('###################################################################')
    print(' Test Model and check accuracy 1 ')
    print('###################################################################')
    prediction = tf.argmax(logits, 1)

    is_correct = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    Y_one_hot = np.eye(nb_classes)[test_labels]

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess,
                feed_dict={X: test_images,Y: Y_one_hot, keep_prob: 1}))

    print('###################################################################')
    print(' Test Model and check accuracy 2 ')
    print('###################################################################')

    test_x_data = test_images    # 10000 X 784
    test_t_data = test_labels    # 10000 X 10
    Y_one_hot = np.eye(nb_classes)[test_t_data]

    accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, Y: Y_one_hot})
    print("\nAccuracy = ", accuracy_val)
