# Lab 3 Minimizing Cost
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(-5000000000000000000.0)

# Linear model
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, W_val, cost_val, hy_val = sess.run([train, W, cost, hypothesis])
        print("Step:", format(step, ',d'), "Cost:", cost_val, "W:", W_val, "hypothesis:", hy_val)