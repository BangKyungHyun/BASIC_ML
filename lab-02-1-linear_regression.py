import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# XW+b를 이용하여 만든 예측
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 최적값
#train = optimizer.minimize(cost)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(200000001):
    sess.run(train)
    
    if step % 100000 == 0:
        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        #cost_val, x_train_val, W_val, b_val, y_train_val, hy_val = sess.run([cost, x_train, W, b, y_train, hypothesis])
        cost_val, W_val, b_val,  hy_val = sess.run([cost, W, b,  hypothesis])

        #print(nowDatetime,"Step:", format(step,',d'), "Cost:", cost_val, "x_train:", x_train_val, "W:", W_val, "B:", b_val, "y_train:", y_train_val, "hypothesis:", hy_val)
        print(nowDatetime,"Step:", format(step,',d'), "Cost:", cost_val, "W:", W_val, "B:", b_val,  "hypothesis:", hy_val)
