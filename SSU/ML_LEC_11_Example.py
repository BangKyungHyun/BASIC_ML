import numpy as np

# f: 미분하고자 하는 다변수 함수
# x : 모든 변수를 포함하고 있는 numpy 객체 (배열, 행렬)
#     x,y 변수가 2개 이면 2 입력, x,y,z 변수가 3개면 3 입력
def numerical_derivative(f, x):  # 수치미분 debug version

    # f : 미분하려고 하는 다변수 함수; f(x,y) = 2x + 3xy + y^3
    # x : 모든 변수를 포함하고 있어야 함 (ndarray)

    delta_x = 1e-4

    grad = np.zeros_like(x)  # 계산된 수치미분 값 저장 변수, 입력 x 만큼 초기화함
    print("debug 1. initial input variable =", x)
    print("debug 2. initial grad =", grad)
    print("=======================================")

    # iterator를 이용해서 입력변수 x에 대해 편미분을 수행
    # 모든 입력변수에 대해 편미분하기 위해 iterator 획득
    # multi_index : iterator생성 후 반복할 때 행렬처럼 (row, column) 형태의 multi_index 형태로 동작
    # readwrite : iterator 을 read / write 형태로 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    print("debug 3. it = ", it)

    # x에 대해서 수치미분 계산하고 다음루프 때 y에 대해서 수치미분 계산
    while not it.finished:    # 변수의 갯수만큼 반복
        idx = it.multi_index  # iterator의 현재 index를 tuple로 추출

        print("debug 4. idx = ", idx, ", x[idx] = ", x[idx])

        # 현재 칸의 값을 어딘가에 잠시 저장
        tmp_val = x[idx]      # numpy 타입은 mutable(변하기 쉬운) 이므로 원래 값 보관

        print("debug 5. tmp_val = ", tmp_val)

        x[idx] = float(tmp_val) + delta_x

        print("debug 6. x[idx] = ", x[idx])

        fx1 = f(x)  # f(x+delta_x)

        print("debug 7. fx1 = ", fx1, 'f(x) =', f(x))

        x[idx] = tmp_val - delta_x
        print("debug 8. x[idx] = ", x[idx])

        fx2 = f(x)  # f(x-delta_x)
        print("debug 9. fx2 = ", fx2, 'f(x) =', f(x))

        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        print("debug 10. grad[idx] = ", grad[idx])
        print("debug 11. grad = ", grad)

        # 데이터 원상복구
        x[idx] = tmp_val
        print("debug 12. x[idx] = ", x[idx])
        print("=======================================")
        it.iternext()

    return grad

# 입력변수 1 개인 함수 f(x) = x**2
def func1(input_obj):
    x = input_obj[0]

    return x ** 2

# x = 3.0 에서의 편미분 값
numerical_derivative(func1, np.array([3.0]))

'''
debug 1. initial input variable = [3.]
debug 2. initial grad = [0.]
=======================================
debug 3. it =  <numpy.nditer object at 0x0000014BA7F66F30>
debug 4. idx =  (0,) , x[idx] =  3.0
debug 5. tmp_val =  3.0
debug 6. x[idx] =  3.0001
debug 7. fx1 =  9.000600010000001 f(x) = 9.000600010000001
debug 8. x[idx] =  2.9999
debug 9. fx2 =  8.999400009999999 f(x) = 8.999400009999999
debug 10. grad[idx] =  6.000000000012662
debug 11. grad =  [6.]
debug 12. x[idx] =  3.0
'''

# 입력변수 2 개인 함수 f(x, y) = 2x + 3xy + y^3
def func1(input_obj):
    x = input_obj[0]
    y = input_obj[1]

    print("debug ***. x = ", x, 'y = ', y)
    return (2 * x + 3 * x * y + np.power(y, 3))


# (x,y) = (1.0, 2.0) 에서의 편미분 값
input = np.array([1.0, 2.0])

numerical_derivative(func1, input)

'''
=======================================
debug 1. initial input variable = [1. 2.]
debug 2. initial grad = [0. 0.]
=======================================
debug 3. it =  <numpy.nditer object at 0x0000014BA7F66F30>
debug 4. idx =  (0,) , x[idx] =  1.0
debug 5. tmp_val =  1.0
debug 6. x[idx] =  1.0001
debug 7. fx1 =  16.000799999999998 f(x) = 16.000799999999998
debug 8. x[idx] =  0.9999
debug 9. fx2 =  15.9992 f(x) = 15.9992
debug 10. grad[idx] =  7.999999999990237
debug 11. grad =  [8. 0.]
debug 12. x[idx] =  1.0
=======================================
debug 4. idx =  (1,) , x[idx] =  2.0
debug 5. tmp_val =  2.0
debug 6. x[idx] =  2.0001
debug 7. fx1 =  16.001500060001003 f(x) = 16.001500060001003
debug 8. x[idx] =  1.9999
debug 9. fx2 =  15.998500059999 f(x) = 15.998500059999
debug 10. grad[idx] =  15.000000010019221
debug 11. grad =  [ 8.         15.00000001]
debug 12. x[idx] =  2.0
'''

# 입력변수 4 개인 함수
# f(w,x,y,z) = wx + xyz + 3w + zy^2
# input_obj 는 행렬
def func1(input_obj):
    w = input_obj[0, 0]
    x = input_obj[0, 1]
    y = input_obj[1, 0]
    z = input_obj[1, 1]

    return (w * x + x * y * z + 3 * w + z * np.power(y, 2))

# 입력을 2X2 행렬로 구성함
input = np.array([[1.0, 2.0], [3.0, 4.0]])

numerical_derivative(func1, input)

'''
=======================================
debug 1. initial input variable = [[1. 2.] [3. 4.]]
debug 2. initial grad = [[0. 0.] [0. 0.]]
=======================================
debug 3. it =  <numpy.nditer object at 0x0000014BA7F66F30>
debug 4. idx =  (0, 0) , x[idx] =  1.0
debug 5. tmp_val =  1.0
debug 6. x[idx] =  1.0001
debug 7. fx1 =  65.0005 f(x) = 65.0005
debug 8. x[idx] =  0.9999
debug 9. fx2 =  64.9995 f(x) = 64.9995
debug 10. grad[idx] =  5.000000000023874
debug 11. grad =  [[5. 0.] [0. 0.]]
debug 12. x[idx] =  1.0
=======================================
debug 4. idx =  (0, 1) , x[idx] =  2.0
debug 5. tmp_val =  2.0
debug 6. x[idx] =  2.0001
debug 7. fx1 =  65.0013 f(x) = 65.0013
debug 8. x[idx] =  1.9999
debug 9. fx2 =  64.9987 f(x) = 64.9987
debug 10. grad[idx] =  13.00000000000523
debug 11. grad =  [[ 5. 13.] [ 0.  0.]]
debug 12. x[idx] =  2.0
=======================================
debug 4. idx =  (1, 0) , x[idx] =  3.0
debug 5. tmp_val =  3.0
debug 6. x[idx] =  3.0001
debug 7. fx1 =  65.00320004000001 f(x) = 65.00320004000001
debug 8. x[idx] =  2.9999
debug 9. fx2 =  64.99680004 f(x) = 64.99680004
debug 10. grad[idx] =  32.00000000006753
debug 11. grad =  [[ 5. 13.] [32.  0.]]
debug 12. x[idx] =  3.0
=======================================
debug 4. idx =  (1, 1) , x[idx] =  4.0
debug 5. tmp_val =  4.0
debug 6. x[idx] =  4.0001
debug 7. fx1 =  65.0015 f(x) = 65.0015
debug 8. x[idx] =  3.9999
debug 9. fx2 =  64.99849999999999 f(x) = 64.99849999999999
debug 10. grad[idx] =  15.000000000000568
debug 11. grad =  [[ 5. 13.] [32. 15.]]
debug 12. x[idx] =  4.0
=======================================

Process finished with exit code 0

'''
