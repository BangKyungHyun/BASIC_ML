import torch

# 파이토치에서 AutoGrad라는 자동 미분 기능을 제공합니다.
# 이를 위해서 파이토치는 requires_grad 속성이 True인 텐서의 연산을 추적하기 위한
# 계산 그래프를 구축하고, backward 함수가 호출되면
# 이 그래프를 따라서 미분을 자동으로 수행하고 계산된 그래디언트를 채워놓습니다.

# 텐서의 requires_grad 속성을 True로 만들 수 있습니다. 해당 속성의 디폴트 값은 False 입니다.
x = torch.FloatTensor([[1,2],[3,4]]).requires_grad_(True)
print("x =", x)
# x = tensor([[1., 2.], [3., 4.]], requires_grad=True)

# 이렇게 requires_grad 속성이 True인 텐서가 있을 때,
# 이 텐서가 들어간 연산의 결과가 담긴 텐서도 자동으로 requires_grad 속성 값을 True로 갖게 됩니다.
# 그럼 다음 코드와 같이 여러가지 연산을 수행하였을 때, 결과 텐서들은 모두 requires_grad 속성 값을 True로 갖게 됩니다.
x1 = x + 2
x2 = x - 2
x3 = x1 * x2
y = x3.sum()

print("x1 =", x1)
# x1 = tensor([[3., 4.],[ 5., 6.]], grad_fn=<AddBackward0>)
print("x2 =", x2)
# x2 = tensor([[-1.,0.],[ 1.,  2.]], grad_fn=<SubBackward0>)
print("x3 =", x3)
# x3 = tensor([[-3.,0.],[ 5., 12.]], grad_fn=<MulBackward0>)
print("y =", y)
# y = tensor(14., grad_fn=<SumBackward0>)

# 앞서 코드의 실행 결과에서 눈여겨 보아야 할 점은, 생성된 결과 텐서들이 모두 grad_fn 속성을 갖는다는 점입니다.
# 예를 들어 텐서 x1이 덧셈 연산의 결과물이기 때문에, x1의 grad_fn 속성은 AddBackward0 임을 볼 수 있습니다.
# 텐서 y는 sum 함수를 썼으므로 스칼라scalar 값이 되었습니다. 그럼 여기세 다음과 같이 backward 함수를 호출합니다.

################################################################################
# 역전파 시키기
################################################################################

y.backward()
print('y.backward() = ', y.backward)
# y.backward =  <bound method Tensor.backward of tensor(14., grad_fn=<SumBackward0>)>

# 그럼 앞서 x, x1, x2, x3, y 모두 grad 속성에 그래디언트 값이 계산되어 저장되었을 것입니다. 이것을 수식으로 나타내면 다음과 같습니다.

#     | x(1,1), x(1,2) |
# x = |                |
#     | x(2,1)  x(2,2) |

# x1 = x + 2
# x2 = x - 2
# x3 = (x + 2)(x - 2) = x2 -4
# y = sum(x3) = x3(1,1)+x3(1,2)+x3(2,1)+x3(2,2)

# 이렇게 구한 y를 다시 x로 미분하면 다음과 같습니다. ==> y 도함수 : x2 - 4 = 2x

#          |    ay      ay      |
#          |   -------  ------- |
# x.grad = |   ax(1,1)  ax(1,2) |
#          |     ay      ay     |
#          |   -------  ------- |
#          | ax(2,1)  ax(2,2)   |
#
# dy
# -- = 2x
# dx

# 그럼 실제 파이토치로 미분한 값과 같은지 비교해보도록 하겠습니다.

print("x.grad = ",x.grad)
# x.grad =  tensor([[2., 4.], [6., 8.]])

print("x = ",x)
# x =  tensor([[1., 2.], [3., 4.]], requires_grad=True)

#print("x3.numpy() = ",x3.numpy())

# Traceback (most recent call last):
#  File "C:\Users\KBDS\PycharmProjects\BASIC_ML\pytorch\Practice_06_01_auto_grad.py", line 32, in <module>
#    print("x3.numpy() = ",x3.numpy())
# RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# Grad가 필요한 Tensor에서 numpy()를 호출할 수 없습니다. 대신 tensor.detach().numpy()를 사용하세요.

# 텐서는 .numpy() 메서드(method)를 호출하여 NumPy 배열로 변환할 수 있습니다.
print("x3.detach_().numpy() = ",x3.detach_().numpy())
# x3.detach_().numpy() =  [[-3.  0.]
#  [ 5. 12.]]