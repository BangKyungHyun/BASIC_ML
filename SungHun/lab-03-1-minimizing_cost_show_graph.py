import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
print(W)
# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_history = []
cost_history = []

# Launch the graph in a session.
with tf.Session() as sess:
    for i in range(-300, 500):
        # 가중치를 입력했을 때 코스트값의 변화를 표현하는 공식
        # y축: cost , X축 : 가중치(W)
        feed_W = i * 0.01
        curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})  # W에 feed_W값이 대입됨

        print("feed_W:=", feed_W, "curr_W:", curr_W, "X:", X, "Y:", Y, "Curr_cost:", curr_cost)

        W_history.append(curr_W)
        cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)
plt.show()
