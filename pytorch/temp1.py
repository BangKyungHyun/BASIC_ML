numpy.random.choice(a, size=None, replace=True, p=None)

a: 1
차원
배열
또는
정수(정수인
경우, np.arange(a)
와
같은
배열
생성)
size: 정수
또는
튜플(튜플인
경우, 행렬로
리턴됨.(m, n, k) -> m * n * k), optional
replace: 중복
허용
여부, boolean, optional
p: 1
차원
배열, 각
데이터가
선택될
확률, optional

numpy.random.choice(5, 3, True)
- 0
이상
5
미만인
정수
중
3
개를
출력한다.(중복
허용)



numpy.random.choice(5, 3, False)
- 0
이상
5
미만인
정수
중
3
개를
출력한다.(중복
비허용)