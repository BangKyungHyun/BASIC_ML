# Lab 4 Multi-variable linear regression
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math

def var(data):
     avg = sum(data) / len(data)
     total = 0
     for i in data:
         total += (avg - i) ** 2  # 편차 (평균과 차이의 제곱)
     return total / len(data)     # 분산 (편차/데이터 갯수)

def std(data):
    return math.sqrt(var(data))

a = [72, 61, 91, 31, 45]

print(sum(a) / len(a)) # 평균 출력
print(var(a))          # 분산 출력  variance
print(std(a))          # 표준편차 출력 standard deviation
