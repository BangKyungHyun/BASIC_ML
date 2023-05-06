# Lab 3 Minimizing Cost
import tensorflow as tf
import datetime
tf.random.set_seed(777)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random.normal([1]), name="weight")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 수동으로 W 계산
# minimize: Gradient Descent using derivative: W -= learning_rate * derivative
# 최소화: 미분함수를 사용한 경사하강법: W -= learning_rate * 미분함수
learning_rate = 1e-5
gradient = tf.reduce_mean((W * X - Y) * X)  # 기울기
descent = W - learning_rate * gradient      # 가중치에서 러닝레이트 * 경사각을 빼줌
update = W.assign(descent)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(2000001):
        _, cost_val, W_val, hy_val = sess.run(
            [update, cost, W, hypothesis], feed_dict={X: x_data, Y: y_data}
        )

        if step % 100000 == 0:
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            #print(step, cost_val, W_val)
            print(nowDatetime,"Step:", format(step,',d'), "Cost:", cost_val, "W:", W_val, "hypothesis:", hy_val)