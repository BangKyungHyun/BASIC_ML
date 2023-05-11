# Lab 4 Multi-variable linear regression

import tensorflow as tf
import tensorflow.compat.v1 as tf
import datetime
tf.disable_v2_behavior()

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis - matrix multiplication(행렬곱 함수)
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2000000001):

    cost_val, w_val, b_val, hy_val, _ = sess.run(
        [cost, W, b, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 100000 == 0:

        #print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        print(nowDatetime,"\nStep:", format(step,',d'), "\nw=\n", w_val, "\nb=", b_val,
              "Cost:", cost_val,"\nPrediction:\n", hy_val)