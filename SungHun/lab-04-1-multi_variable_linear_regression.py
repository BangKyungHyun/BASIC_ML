# Lab 4 Multi-variable linear regression
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime

x1_data = [73., 93., 89., 96., 73., 36]
x2_data = [80., 88., 91., 98., 66.,33]
x3_data = [75., 93., 90., 100., 70., 35]

y_data  = [152., 185., 180., 196., 142.,72]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b  = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2000000001):
     cost_val, hy_val, w1_val, w2_val, w3_val, b_val, _ = sess.run([cost, hypothesis, w1, w2, w3, b, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
     if step % 100000 == 0:
        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,"Step:", format(step,',d'), "w1 =", w1_val, "w2 =", w2_val, "w3 =", w3_val, "b =", b_val,
              "\nCost:", cost_val,"Prediction:", hy_val)