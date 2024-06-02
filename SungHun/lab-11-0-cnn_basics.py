#https://excelsior-cjh.tistory.com/79

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

'''
sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)

print('1부터 9까지의 숫자로 이루어진 matrix를 시각화 해보자')
print('1-1. image.shape =',image.shape)
#1. image.shape =  (1, 3, 3, 1)

plt.imshow(image.reshape(3,3), cmap='gray')
plt.show()

# 9*1 형태로 출력
plt.imshow(image.reshape(9,1), cmap='gray')
plt.show()

# 1*9 형태로 출력
plt.imshow(image.reshape(1,9), cmap='gray')
plt.show()

print("1-2. imag:\n", image)

'''
'''
imag:
 [[[[1.]
   [2.]
   [3.]]

  [[4.]
   [5.]
   [6.]]

  [[7.]
   [8.]
   [9.]]]]
'''
'''
print("2-0. Image : 1,3,3,1 Filter(kernel) : 2,2,1,1 Stride : 1*1 Padding : VALID")
print("1 filter (2,2,1,1) with padding: VALID")
print("입력 3*3 이미지를 2*2 필터로 1*1 Stride 만큼 이동하면 ")
print("(N-F)/Stride +1 공식에 의해 2*2 출력 이미지로 변경 ")

print("2-1. image.shape =", image.shape)
# 2-1. image.shape = (1, 3, 3, 1)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("2-2. weight.shape =", weight.shape)
# 2-2. weight.shape = (2, 2, 1, 1) => 필터(커널)크기 = 2*2, 색깔 = 1, 필터갯수 = 1

# 옆으로 아래로 1칸씩 움직이겠다.
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')

conv2d_img = conv2d.eval()
print("2-3. conv2d_img.shape", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print("2-4. one_img.reshape(2,2)\n",one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
plt.show()

#print("imag:\n", image)
print("3-0. Image : 1,3,3,1 Filter(kernel) : 2,2,1,1 Stride : 1*1 Padding : SAME")
print("1 filter (2,2,1,1) with padding:SAME")
print("Padding : same을 사용하여 입력과 출력의 크기를 같도록 0값으로 padding함")
print("입력 3*3 이미지를 2*2 필터로 1*1 Stride 만큼 이동후에도 3*3 출력이미지가 되도록 ")
print("0 값을 Padding함 ")

print("3-1. image.shape =", image.shape)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("3-2. weight.shape =", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print("3-3. conv2d_img.shape =", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print("3-4. one_img.reshape(3,3)\n",one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
plt.show()

#print("imag:\n", image)
print("4-0. Image : 1,3,3,1 Filter(kernel) : 2,2,1,3 Stride : 1*1 Padding : SAME")
print("필터를 여러개 쓰기 위해서는 필터의 개수를 의미하는 weight.shape=(a,b,c,d)중 d의 값을 늘려주면 된다.")

print("4-1. image.shape =", image.shape)

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print("4-2. weight.shape =", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print("4-3. conv2d_img.shape =", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print("4-4. one_img.reshape(3,3)\n",one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
plt.show()

print("5-0. MAX-POOLING padding='VALID'")
print("1 filter (2,2,1,1) with padding: VALID")
print("입력 3*3 이미지를 2*2 필터로 1*1 Stride 만큼 이동하면 ")
print("(N-F)/Stride +1 공식에 의해 2*2 출력 이미지로 변경 ")
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
print("5-1. pool.shape =", pool.shape)
print("5-2. pool.eval() =", pool.eval())
'''
'''
5-1. pool.shape = (1, 1, 1, 1)
5-2. pool.eval() = [[[[4.]]]]
'''
'''
print("6-0. MAX-POOLING padding='SAME'")

image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print("6-1. pool.shape =", pool.shape)
print("6-2. pool.eval() =", pool.eval())
'''
'''
6-1. pool.shape = (1, 2, 2, 1)
6-2. pool.eval() = [[[[4.]
   [3.]]

  [[2.]
   [1.]]]]
'''


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.
test_images  = test_images  / 255.

#나는 같은 문제에 부딪쳤고 modelis 에 사용된 기본 데이터 유형 float32은 numpyis 의
#데이터 유형이라고 가정하고 해당 유형 float64을 from_tensor_slices유지합니다.
#이 문제를 해결하려면 코드를 변경하세요.
#.astype(np.float32) 추가함 (2022-09-08)
img = train_images[0].reshape([28,28]).astype(np.float32)

plt.imshow(img, cmap='gray')
plt.show()

sess = tf.InteractiveSession()

print('###################################################################')
print('# 신경망 모델 구성')
print('###################################################################')
print('# 기존 모델에서는 입력 값을 28x28 하나의 차원으로 구성하였으나,')
print('# CNN 모델을 사용하기 위해 2차원 평면과 특성치의 형태를 갖는 구조로 만듭니다.')

img = img.reshape([-1,28,28,1])
print('img.shape  =',img.shape)
# img.shape  = (1, 28, 28, 1)

print('###################################################################')
print('MNIST Convolution Layer')
print('###################################################################')

print("W1 [3 3 1 5] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 5: 필터 갯수 ")
print("입력 28*28 이미지를 3*3 필터로 2*2 Stride 만큼 이동하면 ")
print("(N-F)/Stride +1 공식에 의해 (28-3)/2+1 =13+1 =14에 의해 ")
print("14*14 출력 이미지가 필터갯수 만큼 5개 생성 ")
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))

print(" 옆으로 아래로 2칸씩 움직이겠다.")
print(" stride가 1x1일때는 입력과 출력이 같지만, 2x2일때는 반으로 줄어들게 된다.(출력은 14x14)")
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
print("conv2d =", conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
plt.show()

print('###################################################################')
print('MNIST Max Pooling')
print('###################################################################')

print("입력 14*14 이미지를 2*2 필터로 2*2 Stride 만큼 이동하면 ")
print("(N-F)/Stride +1 공식에 의해 (14-2)/2+1 =6+1 =7에 의해 ")
print("7*7 출력 이미지가 필터갯수 만큼 5개 생성 ")
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
print('Pool =',pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.show()