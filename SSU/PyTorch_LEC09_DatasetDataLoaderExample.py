import torch

###############################################################################
# Dataset 정의
###############################################################################
x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.x_train.shape[0]

###############################################################################
# Dataset 인스턴스 / DataLoader 인스턴스 생성
# 중요 : batch_size에서 의해서 학습하는 텐서의 양이 결정됨 
#       예제에서는 batch_size 가 3이므로 x가 1,2,3 y는 3,4,5가 학습됨    
###############################################################################

dataset = CustomDataset(x_train, y_train)

train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)

total_batch = len(train_loader)

print(total_batch)

###############################################################################
# 신경망 모델 구축
###############################################################################

from torch import nn

class MyLinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)

        return prediction

###############################################################################
# 모델 생성
###############################################################################

model = MyLinearRegressionModel()

###############################################################################
# 손실함수 및 옵티마이저 설정
###############################################################################

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for epoch in range(2):

    for idx, batch_data in enumerate(train_loader):

        x_train_batch, y_train_batch = batch_data

        output_batch = model(x_train_batch)

        print('==============================================')
        print('epoch =', epoch+1, ', batch_idx =', idx+1, ',', len(x_train_batch), len(y_train_batch), len(output_batch), len(batch_data) )
        print('x_train_batch = \n ', x_train_batch)
        print('y_train_batch = \n', y_train_batch)
        print('output_batch = \n', output_batch)
        print('batch_data = \n', batch_data)
        print('==============================================')

        # epoch = 2 , batch_idx = 2 , 3 3 3 2
        # x_train_batch =
        #   tensor([[4.],
        #         [5.],
        #         [6.]])
        # y_train_batch =
        #  tensor([[6.],
        #         [7.],
        #         [8.]])
        # output_batch =
        #  tensor([[3.8582],
        #         [4.7478],
        #         [5.6375]], grad_fn=<AddmmBackward0>)
        # batch_data =
        #  [tensor([[4.],
        #         [5.],
        #         [6.]]), tensor([[6.],
        #         [7.],
        #         [8.]])]

        loss = loss_function(output_batch, y_train_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

