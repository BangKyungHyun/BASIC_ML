import tensorflow as tf

# 상수 노드 정의
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')
c = tf.constant([ [1.0, 2.0], [3.0, 4.0] ])

print(a)
print(a+b)
print(c)


# 세션 (session) 을 만들고 노드간의 텐서 연산 실행
sess = tf.Session()

print(sess.run([a, b]))
print(sess.run(c))
print(sess.run([a+b]))
print(sess.run(c+1.0))  # broadcast 수행

# 세션 close
sess.close()
