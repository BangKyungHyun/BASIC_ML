import torch

x = torch.FloatTensor([[1,2],[3,4]]).requires_grad_(True)

x1 = x + 2
x2 = x - 2
x3 = x1 * x2
y = x3.sum()

print("x1 =", x1)
# x1 = tensor([[3., 4.],
# [5., 6.]], grad_fn=<AddBackward0>)
print("x2 =", x2)
# x2 = tensor([[-1.,  0.],
#       [ 1.,  2.]], grad_fn=<SubBackward0>)

print("x3 =", x3)
# x3 = tensor([[-3.,  0.],
#        [ 5., 12.]], grad_fn=<MulBackward0>)

print("y =", y)
# y = tensor(14., grad_fn=<SumBackward0>)

y.backward

print("x.grad = ",x.grad)
# x.grad =  None
print("x = ",x)
# x =  tensor([[1., 2.],
#        [3., 4.]], requires_grad=True)

print("x3.numpy() = ",x3.numpy())

# Traceback (most recent call last):
#  File "C:\Users\KBDS\PycharmProjects\BASIC_ML\pytorch\Practice_06_01_auto_grad.py", line 32, in <module>
#    print("x3.numpy() = ",x3.numpy())
#RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

print("x3.detach_().numpy() = ",x3.detach_().numpy())
# x3.detach_().numpy() =  [[-3.  0.]
#  [ 5. 12.]]
