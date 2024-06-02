# Lab 4 Multi-variable linear regression
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime
import numpy as np

def my_func1(x):
    print('11111')

    print ('x= ',x**2)

    return x**2

def my_func2(x):
    print('11111')

    print ('x= ',3*x*(np.exp(x)))

    return 3*x*(np.exp(x))

def numerical_derivative(f,x):

    print('22222')
    print('x =', x)
    print('f =', f)

    delta_x = 1e-4

    print('33333')
    print('delta_x =', delta_x)

#    print (' 1= ,',(f(x+delta_x) - f(x-delta_x))/ (2*delta_x))
    return (f(x+delta_x) - f(x-delta_x))/ (2*delta_x)

print('44444')

result = numerical_derivative(my_func1,3)

print('55555')
print ('result 1 =', result)

print('66666')
result = numerical_derivative(my_func2,2)

print('77777')
print ('result 2 =', result)


print('88888')
'''
def my_func1(x):
    print('11111')

    print ('x= ',x**2)

    return x**2

def numerical_derivative(my_func1,x):

    print('22222')
    print('x =', x)
    print('f =', my_func1)

    delta_x = 1e-4

    print('33333')

    print (' 1= ,',(my_func1(x+delta_x) - my_func1(x-delta_x))/ (2*delta_x))
    return (my_func1(x+delta_x) - my_func1(x-delta_x))/ (2*delta_x)

print('44444')

result = numerical_derivative(my_func1,3)

print('55555')
print ('result =', result)

print('66666')

'''
