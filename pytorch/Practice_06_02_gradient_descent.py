import torch
import torch.nn.functional as F
# 목표 텐서 생성
target = torch.FloatTensor([[.1,.2,.3],
                            [.4,.5,.6],
                            [.7,.8,.9]])

# 랜덤값을 갖는 텐서 하나를 생성
# [중요] 텐서의 requires_grad 속성이 True가 되도록 설정해 주어야 함
x = torch.rand_like(target)
x.requires_grad = True

# 랜덤 생성한 텐서의 값을 출력
print("x =", x)
# x = tensor([[0.9335, 0.3228, 0.8134],
#        [0.4263, 0.2838, 0.1215],
#        [0.7253, 0.2697, 0.4487]], requires_grad=True)

# 두 텐서 사이의 비용값을 계산
loss = F.mse_loss(x, target)
print("loss =", loss)
# loss = tensor(0.1928, grad_fn=<MseLossBackward0>)

threshold = 1e-10   # 임계치
learning_rate = 1.
iter_cnt = 0
# while 반복문을 사용하여 두 텐서 값의 차이가 threshold의 값보다 작아질 때 까지
# 미분 및 경사하강법을 반복 수행

while loss > threshold:
    iter_cnt += 1
    # backward 함수를 통해 편미분을 수행한다는 것인데 편미분을 통해 얻어진 기울기들이 x.grad에 자동
    # 저장되고 이 값을 활용하여 경사하강법을 수행한다.
    # backward를 호출하기 위한 텐서의 크기는 스칼라이어야 함
    # 만약 스칼라가 아닌 경우에 backward를 호출하면 파이토치는 오류를 발생시킴
    loss.backward() # 기울기 계산
    x = x - learning_rate * x.grad

    x.detach_()
    x.requires_grad_(True)

    loss = F.mse_loss(x, target)

    print('%d-th Loss : %.4e' % (iter_cnt, loss))
    print("x = ",x )

