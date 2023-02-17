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

print(x.reshape(12)) # 12 = 3 * 2 * 2
print(x.reshape(-1))

print(x.reshape(3, 4)) # 3 * 4 = 3 * 2 * 2
print(x.reshape(3, -1))

print(x.reshape(3, 1, 4))
print(x.reshape(-1, 1, 4))

print(x.reshape(3, 2, 2, 1))

#You can use 'view()' instead of 'reshape()' in some cases.

#https://pytorch.org/docs/stable/tensor_view.html
#https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view

#squeeze: Remove dimension which has only one element

x = torch.FloatTensor([[[1, 2],
                        [3, 4]]])
print(x.size())

# Remove any dimension which has only one element.
print(x.squeeze())
print(x.squeeze().size())

# Remove certain dimension, if it has only one element. If it is not, there would be no change.

print(x.squeeze(0).size())
print(x.squeeze(1).size())

# unsqueeze: Insert dimension at certain index.
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(x.size())
print(x.unsqueeze(2))
print(x.unsqueeze(-1))
print(x.reshape(2, 2, -1))