import torch

# Tensor Shaping
# reshape: Change Tensor Shape

x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])

print(x.size())
#torch.Size([3, 2, 2])

print(x.reshape(12)) # 12 = 3 * 2 * 2
#tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.])

print(x.reshape(-1))
#tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.])

print(x.reshape(3, 4)) # 3 * 4 = 3 * 2 * 2
#tensor([[ 1.,  2.,  3.,  4.],
#        [ 5.,  6.,  7.,  8.],
#        [ 9., 10., 11., 12.]])
print(x.reshape(3, -1))
#tensor([[ 1.,  2.,  3.,  4.],
#        [ 5.,  6.,  7.,  8.],
#        [ 9., 10., 11., 12.]])
print(x.reshape(3, 1, 4))
#tensor([[[ 1.,  2.,  3.,  4.]],
#
#        [[ 5.,  6.,  7.,  8.]],
#
#        [[ 9., 10., 11., 12.]]])
print(x.reshape(-1, 1, 4))
#tensor([[[ 1.,  2.,  3.,  4.]],
#
#        [[ 5.,  6.,  7.,  8.]],
#
#        [[ 9., 10., 11., 12.]]])

print(x.reshape(3, 2, 2, 1))
# tensor([[[[ 1.],
#           [ 2.]],
#          [[ 3.],
#           [ 4.]]],
#         [[[ 5.],
#           [ 6.]],
#          [[ 7.],
#           [ 8.]]],
#         [[[ 9.],
#           [10.]],
#          [[11.],
#           [12.]]]])

#You can use 'view()' instead of 'reshape()' in some cases.

#https://pytorch.org/docs/stable/tensor_view.html
#https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view

#squeeze: Remove dimension which has only one element

x = torch.FloatTensor([[[1, 2],
                        [3, 4]]])
print(x.size())
#torch.Size([1, 2, 2])

# Remove any dimension which has only one element.
# 요소가 하나만 있는 차원을 제거하십시오
# np.squeeze 함수 기본 사용법
#np.squeeze 함수의 사용법은 np.squeeze(배열) 혹은 배열.squeeze() 형태로 지정해주시면 간단히 완료됩니다.
#a 배열의 바깥쪽 추가 axis가 제거되고 2차원 배열로 변환이 되었습니다.
print(x.squeeze())
#tensor([[1., 2.],
#        [3., 4.]])
print(x.squeeze().size())

# Remove certain dimension, if it has only one element. If it is not, there would be no change.
# 요소가 하나만 있는 경우 특정 차원을 제거합니다. 그렇지 않다면 변화가 없을 것 입니다.
print(x.squeeze(0).size())
#torch.Size([2, 2])
print(x.squeeze(1).size())
#torch.Size([2, 2])

# unsqueeze: Insert dimension at certain index.
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(x.size())
#torch.Size([2, 2])
print(x.unsqueeze(2))
#torch.Size([1, 2, 2])
print(x.unsqueeze(-1))
#torch.Size([2, 2])
print(x.reshape(2, 2, -1))
#tensor([[[1.],
#         [2.]],
#        [[3.],
#         [4.]]])
#tensor([[[1.],
#         [2.]],
#        [[3.],
#         [4.]]])
#tensor([[[1.],
#         [2.]],
#        [[3.],
#         [4.]]])