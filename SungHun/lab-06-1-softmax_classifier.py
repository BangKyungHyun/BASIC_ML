import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

y_argmax = tf.argmax(Y,1)

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
matmul = tf.matmul(X, W) + b

# 점수로 환산한 가설
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
print(hypothesis)

# Correct prediction Test model
# Softmax를 통해 나온 결과중 최대값의 인덱스를 얻고자 할 때 사용한다.
prediction = tf.argmax(hypothesis, 1)  # 1 = 가장 안의 배열에서 데이터를 가져온다.

p_one_hot = tf.one_hot(prediction, 3)
print(p_one_hot)

# 추가 : softmax 값을 one-hot encoding 할 수 있도록 argmax함수 사용(2022.07.31)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Cross entropy cost/loss (실제와 가설간의 차이 계산)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val, w_val, m_val, h_val, p_o_val, p_val, c_val, a_val, y_val \
                = sess.run([optimizer, cost, W, matmul, hypothesis, p_one_hot, prediction, is_correct, accuracy, y_argmax],
                           feed_dict={X: x_data, Y: y_data})

            if step % 500 == 0:
                print('========================================')
                print('step =>',step)
                print('cost =>', cost_val)
                print('x_data =>', x_data)
                print('X =>', X)
                print('y_data =>', y_data)
                print('Y =>', Y)

                print('----------------------------------------')
                print('--------------- weight -----------------')
                print('----------------------------------------')
                print('1. weight (X1, X2, X3, X4 변수에 대한 0,1,2 가중치 값)')
                print('2. 상하(Y값): X1, X2, X3, X4 <==> 좌우(X값) : 0, 1, 2 가중치 ')
                print('3. W = tf.Variable(tf.random_normal([4, nb_classes]), name='') 식에 의해 2차원 배열로 만들어진 집합 ')                
                print('4. 첫번째는 무작위로 가중치가 생성되고 이후로는 옵티마이저에 의해 최적화된 가중치가 계속 반영 ')                
                print('')                
                print(w_val)
               
                print('----------------------------------------')
                print('--------------- matmul -----------------')
                print('----------------------------------------')
                print('1. 각 레코드별로 X * W 행렬곱으로 계산 완료 값 + b)')                
                print('2. 상하(Y값): 레코드 갯수만큼 표시 <==> 좌우(X값) : 레코드별로  0, 1, 2 될 점수 ')                
                print('3. matmul = tf.matmul(X, W) + b ')
                print('')                
                print(m_val)

                print('----------------------------------------')
                print('-------------- hyphothesis -------------')
                print('----------------------------------------')
                print('행렬값을 softmax로 변환하여 각 레코드별로 0,1,2 될 확률 합계가 1.0이 되게 함')
                print('hyphothesis = tf.nn.softmax(tf.matmul(X, W) + b) ')
                print('')     
                print(h_val)

                print('----------------------------------------')
                print('--------------- prediction -------------')
                print('----------------------------------------')
                print('hypothesis 결과에서 argmax함수을 통해 최대값의 인덱스를 계산한다.')
                print('prediction = tf.argmax(hypothesis, 1)')
                print('')     
                print(p_val)

                print('----------------------------------------')
                print('---------- prediction -> one hot ---------')
                print('----------------------------------------')
                print('prediction -> prediction one hot으로 표현')
                print('p_one_hot = tf.one_hot(prediction, 3) ')
                print('')
                print(p_o_val)

                print('----------------------------------------')
                print('---------------  Y argmax  -------------')
                print('----------------------------------------')
                print('Y 값을 argmax함수를 통해 최대값의 인덱스를 계산한다.')
                print('y_argmax = tf.argmax(Y, 1)')
                print('')     
                print(y_val)

                print('----------------------------------------')
                print('--------------- is corret  -------------')
                print('----------------------------------------')
                print('prediction 값과 실제 Y 값을 argmax한 값을 비교하여 맞으며 T, 틀리면 F로 표시')
                print('is corret = tf.equal(prediction, tf.argmax(Y, 1))')
                print('')     
                print(c_val)
  
                print('----------------------------------------')
                print('--------------- accurary   -------------')
                print('----------------------------------------')
                print('8개의 레코드에 대한 예측값과 실제값을 비교 후에 맞은 평균값 계산 ')
                print('accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))')
                print('')     
                print(a_val)
  
    print('===========================')
    print('=======Test Data ==========') 
    print('===========================')
        
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))