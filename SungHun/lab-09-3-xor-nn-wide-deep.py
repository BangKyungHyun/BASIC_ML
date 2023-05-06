import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#prediction1 = tf.cast(layer1 > 0.5, dtype=tf.float32)

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
#prediction2 = tf.cast(layer2 > 0.5, dtype=tf.float32)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
#prediction3 = tf.cast(layer3 > 0.5, dtype=tf.float32)

W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(200001):
        _, cost_val, x_val, y_val, w1_val, w2_val, w3_val, b1_val, b2_val, b3_val, layer1_val, layer2_val, layer3_val, h_val, p_val, a_val \
            = sess.run(
            [train, cost, X, Y, W1, W2, W3, b1, b2, b3, layer1, layer2, layer3,
             #prediction1, prediction2, prediction3,
             hypothesis, predicted, accuracy],
            feed_dict={X: x_data, Y: y_data})
        if step % 10000 == 0:
            print('Step =', step, 'cost =', cost_val, '\nx=\n', x_val, '\ny=\n', y_val,
                  '\nw1 =\n', w1_val, '\nb1 =\n', b1_val, '\nlayer1 =\n', layer1_val,
                  '\nw2 =\n', w2_val, '\nb2 =\n', b2_val, '\nlayer2 =\n', layer2_val,
                  '\nw3 =\n', w3_val, '\nb3 =\n', b3_val, '\nlayer3 =\n', layer3_val,
                  '\nhypothsis =\n', h_val, '\npredicted =\n', p_val, '\naccuracy =\n', a_val)

    # Accuracy report
    h, c, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)