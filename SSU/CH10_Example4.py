import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
tf.compat.v1.disable_eager_execution()
import datetime

# read_data_sets() 를 통해 데이터를 객체형태로 받아오고
# 정답(label)은 한 자리 숫자로 저장되어 있기 때문에
# one_hot 옵션을 통해 정답(label) 을 one-hot 인코딩된 형태로 받아옴
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 28*28 2차원의 배열을 784 픽셀의 1차원 배열 형태로 만들기 위해 reshape 한다.
train_images = train_images.reshape([-1, 784])
test_images  = test_images.reshape([-1, 784])

# 원본 데이터는 각 픽셀이 0~255까지의 숫자로 음영의 강도를 표시함
# train, test set 모두 255로 나누어 범위값을 0~1사이로 표준화
train_images = train_images / 255.
test_images  = test_images  / 255.

# 입력노드, 은닉노드, 출력노드, 학습율, 반복횟수, 배치 개수 등 설정
learning_rate = 0.1  # 학습율
num_epochs = 100            # 반복횟수
batch_size = 100      # 한번에 입력으로 주어지는 MNIST 개수

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

W2 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))  # 은닉층 가중치 노드
b2 = tf.Variable(tf.random_normal([hidden_nodes]))               # 은닉층 바이어스 노드

W3 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes])) # 출력층 가중치 노드
b3 = tf.Variable(tf.random_normal([output_nodes]))               # 출력층 바이어스 노드

# 선형회귀 선형회귀 값 Z2 # 은닉층 출력 값 A2, sigmoid 대신 relu 사용
Z2 = tf.matmul(X, W2) + b2
A2 = tf.nn.relu(Z2)

# 출력층 선형회귀  값 Z3, 즉 softmax 에 들어가는 입력 값
Z3 = logits = tf.matmul(A2, W3) + b3
A3 = tf.nn.softmax(Z3)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교함
predicted_val = tf.equal( tf.argmax(A3, 1), tf.argmax(Y, 1) )

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
                      '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    print('###################################################################')
    print(' Test Model and check accuracy 1')
    print('###################################################################')

    prediction = tf.argmax(logits, 1)
    is_correct = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    Y_one_hot = np.eye(nb_classes)[test_labels]

    print('###################################################################')
    print(' Test Model and check accuracy 2')
    print('###################################################################')

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print(nowDatetime, "Accuracy :", accuracy.eval(session=sess,
                feed_dict={X: test_images,Y: Y_one_hot, keep_prob: 1}))
