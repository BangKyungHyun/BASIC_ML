import numpy as np

# 미분 공식 구현
def simple_derivative(f, var):
    delta = 1e-5
    print("f(var + delta) =", f(var + delta))
    print("f(var - delta) =", f(var - delta))
    print("(2 * delta) =", (2 * delta))

    diff_val = (f(var + delta) - f(var - delta)) / (2 * delta)

    print("f(var + delta) =", f(var + delta))
    print("f(var - delta) =", f(var - delta))
    print("(2 * delta) =", (2 * delta))

    return diff_val

# 미분대상 함수
def func1(x):
    print("def func1(x): start")
    return x ** 2
    print("def func1(x): end")

# lambda function을 이용하여 함수 f 로 정의
f = lambda x: func1(x)

ret_val = simple_derivative(f, 3.0)

print(ret_val)
