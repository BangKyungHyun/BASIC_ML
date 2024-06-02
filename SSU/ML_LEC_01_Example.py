# f : 미분할 함수 / x : 미분값을 알고자하는 입력 값

# 수치 미분 구현
def numerical_derivative(f, x):
    # delta_x는 1e-4 ~ 1e-6 정도로 설정하는 것이 좋음

    print("numerical_derivative f 1 = ", f)
    print("numerical_derivative x 1 = ", x)

# Python에서 delta_x가 소수점 8자리 이하로 내려가면 오류가 발생(반올림에서 오류)
# 일반적으로 delta_x를 0.00001 정도로 설정

    delta_x = 1e-4

# 중앙차분방식으로 미분 수행(이 방식이 가장 정확한 값을 도출함)
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)


# 미분하려는 함수 생성 : f(x) = x^2
def my_func(x):
    print("my_func(x) 1 = ", x)
    print("my_func(x) 2 = ", x ** 2)

    return x ** 2

# f'(5)를 구해보자
print("result = numerical_derivative(my_func, 3)")
result = numerical_derivative(my_func, 3)

# 결과확인
print('result = {}'.format(result))  # result = 6.000000000012662
# 근사값이기 때문에 정확히 10으로 나오지는 않음

print('(3.0001 ** 2 - 2.9999 ** 2) / 0.00001*2 = ',(3.0001 ** 2 - 2.9999 ** 2) / (2*0.0001))

'''
numerical_derivative f 1 =  <function my_func at 0x00000179FD4F1E50>
numerical_derivative x 1 =  3
my_func(x) 1 =  3.0001
my_func(x) 2 =  9.000600010000001
my_func(x) 1 =  2.9999
my_func(x) 2 =  8.999400009999999
result = 6.000000000012662
(3.0001 ** 2 - 2.9999 ** 2) / 0.00001*2 =  6.000000000012662
'''
