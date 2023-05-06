import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime

# Try to find value for W and b to compute y_data = x_data * W + b  
# We know that W should be 1 and b should be 0
# But let's TensorFlow figure it out 
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#### Now we can use X and Y in place of x_data and y_data
#### placeholders for a tensor that will be always fed using feed_dict
#### See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Out hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(400000001):
    cost_val, W_val, b_val, hy_val, _ = sess.run([cost, W, b, hypothesis, train],
                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})

    if step % 100000 == 0:

        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        print(nowDatetime, "Step:", format(step,',d'), "Cost:", cost_val, "W:", W_val, "B:", b_val,"hypothesis:", hy_val)

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

for step in range(400000001):
    cost_val, W_val, b_val, hy_val, _ = sess.run([cost, W, b, hypothesis, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 100000 == 0:
        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        print(nowDatetime, "Step:", format(step,',d'), "Cost:", cost_val, "W:", W_val, "B:", b_val,"hypothesis:", hy_val)

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))