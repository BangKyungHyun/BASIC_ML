import torch

################################################################################
# autograd 활성화 시키기 - requires_grad_(True)
# Tensor를 생성하기 위해 사용하는 함수들의 파라미터로 requires_grad=True 를 넘김
################################################################################

x = torch.FloatTensor([[1,2],[3,4]]).requires_grad_(True)
print("x =", x)
# x = tensor([[1., 2.], [3., 4.]], requires_grad=True)

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

################################################################################
# 역전파 시키기
################################################################################

y.backward()
print('y.backward() = ', y.backward)
# y.backward =  <bound method Tensor.backward of tensor(14., grad_fn=<SumBackward0>)>

# x1 = x + 2
# x2 = x - 2
# x3 = (x + 2)(x - 2) = x2 -4
# y = sum(x3) = x3(1,2)+x3(1,2)+x3(2,1)+x3(2,2)
# y 도함수 : x2 - 4 = 2x

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