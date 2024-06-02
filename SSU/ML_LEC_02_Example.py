def numerical_differentiation(f, x, method='central'):
    """
    첫번째 인자 f는 미분하려고 하는 함수
    두번째 인자 x는 x의 값
    세번째 인자 method = 'central'  중앙차분이 디폴트.
                       'forward' 전방차분
                       'backward' 후방차분
    """
    delta_x = 1e-4  # 극한값을 고려. 델타x는 0에 최대한 가까워야함. 여기선 1 * 10^(-4)를 사용

    if method == 'central':
        result = (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)
    elif method == 'forward':
        result = (f(x + delta_x) - f(x)) / delta_x
    elif method == 'backward':
        result = (f(x) - f(x - delta_x)) / delta_x
    else:
        raise ValueError(
            "Method must be either 'central', 'forward', or 'backward'")
    return result

def squared(x):
  return x ** 2

print("중앙차분의 값은 {} 입니다.".format(numerical_differentiation(squared, 3)))
print("전향차분의 값은 {} 입니다.".format(numerical_differentiation(squared, 3, 'forward')))
print("후향차분의 값은 {} 입니다.".format(numerical_differentiation(squared, 3, 'backward')))
