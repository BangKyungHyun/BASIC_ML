# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import pprint
#tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Simple Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

#2D Array

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

# Shape, Rank, Axis
t = tf.constant([1,2,3,4])

tf.shape(t).eval()

t = tf.constant([[1,2],
                 [3,4]])
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

[
    [
        [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
]

#Matmul VS multiply
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
tf.matmul(matrix1, matrix2).eval()

(matrix1*matrix2).eval()

#Watch out broadcasting
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()

# Random values for variable initializations

tf.random_normal([3]).eval()

tf.random_uniform([2]).eval()

tf.random_uniform([2, 3]).eval()

# Reduce Mean/Sum

tf.reduce_mean([1, 2], axis=0).eval()

x = [[1., 2.],
     [3., 4.]]


tf.reduce_mean(x).eval()

tf.reduce_mean(x, axis=0).eval()

array([ 2.,  3.], dtype=float32)

tf.reduce_mean(x, axis=1).eval()

tf.reduce_mean(x, axis=-1).eval()

tf.reduce_sum(x).eval()

tf.reduce_sum(x, axis=0).eval()

tf.reduce_sum(x, axis=-1).eval()

tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()

# Argmax with axis
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()

tf.argmax(x, axis=1).eval()

tf.argmax(x, axis=-1).eval()