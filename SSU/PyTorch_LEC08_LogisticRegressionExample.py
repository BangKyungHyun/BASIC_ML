import numpy as np

loaded_data = np.loadtxt('../data/diabetes.csv', delimiter=',')

x_train_np = loaded_data[ : , 0:-1]
y_train_np = loaded_data[ : , [-1]]

print('loaded_data.shape = ', loaded_data.shape)
print('x_train_np.shape = ', x_train_np.shape)
print('y_train_np.shape = ', y_train_np.shape)

# loaded_data.shape =  (759, 9)
# x_train_np.shape =  (759, 8)
# y_train_np.shape =  (759, 1)

import torch
from torch import nn

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

class MyLogisticRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.logistic_stack(data)

        return prediction

model = MyLogisticRegressionModel()

for param in model.parameters():
    print(param)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

train_loss_list = []
train_accuracy_list = []

nums_epoch = 3

for epoch in range(nums_epoch+1):

    outputs = model(x_train)
    # print('outputs.shape =', outputs.shape)
    # outputs.shape = torch.Size([759, 1])
    #
    # print('outputs =', outputs)
    # outputs = tensor([[0.4057],
    #                   [0.4356],
    #                   [0.3961],
    #                   [0.4704],
    #                   [0.4725],
    #                   [0.4496],
    #                   [0.5423],
    #                   [0.5202],
    #                   [0.3119],
    #                   [0.4530],
    #                   [0.4855],


    loss = loss_function(outputs, y_train)
    # print('loss.shape =', loss.shape)
    # loss.shape = torch.Size([])

    # print('loss =', loss)
    # loss = tensor(0.6997, grad_fn= < BinaryCrossEntropyBackward0 >)

    # print('loss.item() =', loss.item())
    # loss.item() = 0.6791601181030273
    # loss.item() = 0.6749033331871033
    # loss.item() = 0.6710168123245239
    # loss.item() = 0.6674610376358032

    #
    train_loss_list.append(loss.item())
    # print('train_loss_list =', train_loss_list)
    # train_loss_list = [0.6791601181030273]
    # train_loss_list = [0.6791601181030273, 0.6749033331871033]
    # train_loss_list = [0.6791601181030273, 0.6749033331871033,0.6710168123245239]
    # train_loss_list = [0.6791601181030273, 0.6749033331871033,0.6710168123245239, 0.6674610376358032]

    prediction = outputs > 0.5
    # print('prediction.shape =', prediction.shape)
    # prediction.shape = torch.Size([759, 1])

    # print('prediction =', prediction)
    # prediction = tensor([[False],
    #                      [False],
    #                      [False],
    #                      [False],
    #                      [False],

    # print('prediction.float() =', prediction.float())
    # prediction.float() = tensor([[0.],
    #                              [0.],
    #                              [0.],
    #                              [0.],
    #                              [0.],
    #                              [0.],
    #                              [1.],
    #                              [1.],

    correct = (prediction.float() == y_train)
    # print('correct.shape =', correct.shape)
    # correct.shape = torch.Size([759, 1])

    # print('correct =', correct)
    # correct = tensor([[True],
    #                   [False],
    #                   [True],
    #                   [False],

    accuracy = correct.sum().item() / len(correct)
    print('accuracy.shape =', accuracy.shape)
    # AttributeError: 'float' object has no attribute 'shape'

    # print('len(correct) =', len(correct))
    # len(correct) = 759
    # len(correct) = 759
    # len(correct) = 759

    # print('correct.sum() =', correct.sum())
    # correct.sum() = tensor(445)
    # correct.sum() = tensor(453)
    # correct.sum() = tensor(456)
    # correct.sum() = tensor(460)
    #
    # print('correct.sum().item() =', correct.sum().item())
    # correct.sum().item() = 445
    # correct.sum().item() = 453
    # correct.sum().item() = 456
    # correct.sum().item() = 460

    # print('accuracy =', accuracy)
    # accuracy = 0.5862977602108037
    # accuracy = 0.5968379446640316
    # accuracy = 0.6007905138339921
    # accuracy = 0.6060606060606061

    train_accuracy_list.append(accuracy)
    # print('train_accuracy_list=', train_accuracy_list)
    # train_accuracy_list = [0.5862977602108037]
    # train_accuracy_list = [0.5862977602108037, 0.5968379446640316]
    # train_accuracy_list = [0.5862977602108037, 0.5968379446640316, 0.6007905138339921]
    # train_accuracy_list = [0.5862977602108037, 0.5968379446640316, 0.6007905138339921, 0.6060606060606061]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item(), ' accuracy = ', accuracy)

    # epoch =  0  current loss =  0.6791601181030273  accuracy =  0.5862977602108037


# for name, child in model.named_children():
#     for param in child.parameters():
#         print(name, param)
#
# for param in model.parameters():
#     print(param)


import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.legend(loc='best')

plt.show()