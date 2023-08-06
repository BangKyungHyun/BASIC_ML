import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# XW+b를 이용하여 만든 추정값
hypothesis = x_train * W + b

# 실제값(y_train)과 추정값(hypothesis) 사이의 차이(비용)(MSE : 평균제곱오류)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 경사 하강법에 의한 가중치 최적화(학습률 0.1)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(1001):
    # 학습 수행
    sess.run(train)
    
    if step % 10 == 0:
        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        cost_val, W_val, b_val,  hy_val = sess.run([cost, W, b,  hypothesis])
        print(nowDatetime,"Step:", format(step,',d'), "Cost:", cost_val, "W:", W_val, "B:", b_val,  "hypothesis:", hy_val)
