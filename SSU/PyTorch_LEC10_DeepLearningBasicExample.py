import torch

x_train = torch.Tensor([2, 4, 6, 8, 10,12, 14, 16, 18, 20]).view(10, 1)
y_train = torch.Tensor([0, 0, 0, 0, 0,  0, 1,   1,  1,  1]).view(10, 1)

print(x_train.shape, y_train.shape)

from torch import nn

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deeplearning_stack = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.deeplearning_stack(data)
        return prediction

deeplearning_model = MyDeepLearningModel()

for name, child in deeplearning_model.named_children():
    for param in child.parameters():
        print(name, param)

# deeplearning_stack Parameter containing:
# tensor([[-0.2934],
#         [ 0.6282],
#         [ 0.1318],
#         [ 0.4041],
#         [ 0.6585],
#         [ 0.9102],
#         [ 0.1555],
#         [ 0.6406]], requires_grad=True)
# deeplearning_stack Parameter containing:tensor([ 0.5368,  0.1693, -0.2009, -0.0361, -0.7692,  0.6526,  0.5829, -0.1540],requires_grad=True)
# deeplearning_stack Parameter containing:tensor([[-0.3474,  0.0344, -0.2802, -0.1476, -0.3420,  0.2646,  0.3501, -0.0724]],requires_grad=True)
# deeplearning_stack Parameter containing:tensor([0.1789], requires_grad=True)


loss_function = nn.BCELoss()

optimizer = torch.optim.SGD(deeplearning_model.parameters(), lr=1e-2)


nums_epoch = 50000

for epoch in range(nums_epoch+1):

    outputs = deeplearning_model(x_train)

    loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())

for name, child in deeplearning_model.named_children():
    for param in child.parameters():
        print(name, param)

# deeplearning_stack Parameter containing:
# tensor([[-0.5072],
#         [-0.0548],
#         [ 0.0754],
#         [ 0.0199],
#         [ 0.5100],
#         [-0.3087],
#         [-0.3022],
#         [ 0.1464]], requires_grad=True)
# deeplearning_stack Parameter containing: tensor([ 5.3562,  0.5783, -0.7968, -0.2105, -5.3855,  3.2596,  3.1913, -1.5464], requires_grad=True)
# deeplearning_stack Parameter containing: tensor([[-1.7032, -0.1839,  0.2534,  0.0669,  1.7125, -1.0365, -1.0148,  0.4917]], requires_grad=True)
# deeplearning_stack Parameter containing: tensor([-5.9278], requires_grad=True)

deeplearning_model.eval()

test_data = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6,1)

pred = deeplearning_model(test_data)

logical_value = (pred > 0.5).float()

print(pred)

# tensor([[4.4663e-14],
#         [2.1261e-11],
#         [7.2964e-11],
#         [7.8088e-03],
#         [5.2191e-01],
#         [1.0000e+00]], grad_fn=<SigmoidBackward0>)


print(logical_value)

# tensor([[0.],
#         [0.],
#         [0.],
#         [0.],
#         [1.],
#         [1.]])
#
