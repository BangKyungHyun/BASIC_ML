import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")

FEATURE_NUMS = 4        # 입력층으로 들어가는 데이터 개수 feature  ('Open', 'High', 'Low', 'Volume')
SEQ_LENGTH = 5          # 정답을 만들기 위해 필요한 시점 개수 time step
HIDDEN_SIZE = 4         # RNN 계열 계층을 구성하는 hidden state 개수
NUM_LAYERS = 1          # RNN 계열 계층이 몇겹으로 쌓였는지 나타냄
LEARNING_RATE = 1e-3    # 학습율
BATCH_SIZE = 20         # 학습을 위한 배치사이즈 개수

################################################################################
# 데이터 불러오기
################################################################################

import FinanceDataReader as fdr

# 삼성전자 (005930) 주가
df = fdr.DataReader('005930', '2024-01-01', '2024-07-20')

df = df[ ['Open', 'High', 'Low', 'Volume', 'Close'] ]

print('df.tail(10) =', df.tail(10))

print('len(df) =', len(df))

# len(df) = 136
################################################################################
# 데이터 train, test 용 분리하기
################################################################################

# train : test - 70 : 30 분리

SPLIT = int(0.7*len(df))  # train : test = 7 : 3

train_df = df[ :SPLIT ]
test_df = df[ SPLIT: ]

################################################################################
# 변수 정규화
################################################################################

scaler_x = MinMaxScaler()  # feature scaling

train_df.iloc[ : , :-1 ] = scaler_x.fit_transform(train_df.iloc[ : , :-1 ])
test_df.iloc[ : , :-1 ] = scaler_x.fit_transform(test_df.iloc[ : , :-1 ])

scaler_y = MinMaxScaler()  # label scaling

train_df.iloc[ : , -1 ] = scaler_y.fit_transform(train_df.iloc[ : , [-1] ])
test_df.iloc[ : , -1 ] = scaler_y.fit_transform(test_df.iloc[ : , [-1] ])

################################################################################
# 순차적 Numpy Data 만들기
################################################################################

def MakeSeqNumpyData(data, seq_length):
    x_seq_list = []
    y_seq_list = []

    print('len(data) =',len(data))
    print('data.shape =',data.shape)

    print('seq_length =',seq_length)

    # len(data) = 95 --> train
    # data.shape = (95, 5)
    # seq_length = 5

    # len(data) = 41 --> test
    # data.shape = (41, 5)
    # seq_length = 5

    for i in range(len(data) - seq_length):
        x_seq_list.append(data[i:i + seq_length, :-1])
        y_seq_list.append(data[i + seq_length, [-1]])

    x_seq_numpy = np.array(x_seq_list)
    y_seq_numpy = np.array(y_seq_list)

    print('def MakeSeqNumpyData(data, seq_length)')
    print('x_seq_numpy.shape = ',x_seq_numpy.shape)
    print('x_seq_numpy.size = ',x_seq_numpy.size)
    print('y_seq_numpy.shape = ',y_seq_numpy.shape)
    print('y_seq_numpy.size = ',y_seq_numpy.size)

    # def MakeSeqNumpyData(data, seq_length) --> train
    # x_seq_numpy.shape =  (90, 5, 4)
    # x_seq_numpy.size =  1800
    # y_seq_numpy.shape =  (90, 1)
    # y_seq_numpy.size =  90

    # def MakeSeqNumpyData(data, seq_length) --> test
    # x_seq_numpy.shape =  (36, 5, 4)
    # x_seq_numpy.size =  720
    # y_seq_numpy.shape =  (36, 1)
    # y_seq_numpy.size =  36

    return x_seq_numpy, y_seq_numpy

################################################################################
# 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환
################################################################################

x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEQ_LENGTH)

x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEQ_LENGTH)

print(x_train_data.shape, y_train_data.shape)
print(x_test_data.shape, y_test_data.shape)

# (90, 5, 4) (90, 1)  -> train
# (36, 5, 4) (36, 1)  -> test

x_train_tensor = torch.FloatTensor(x_train_data).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train_data).to(DEVICE)

# print('x_train_tensor = ',x_train_tensor)

# x_train_tensor =  tensor([[[0.4853, 0.5571, 0.5435, 0.1544],
#          [0.5074, 0.4857, 0.4565, 0.2505],
#          [0.3309, 0.3786, 0.3913, 0.1165],
#          [0.3750, 0.3643, 0.4130, 0.0326],
#          [0.3971, 0.3929, 0.4130, 0.0281]],
#
#         [[0.5074, 0.4857, 0.4565, 0.2505],
#          [0.3309, 0.3786, 0.3913, 0.1165],
#          [0.3750, 0.3643, 0.4130, 0.0326],
#          [0.3971, 0.3929, 0.4130, 0.0281],
#          [0.4265, 0.4071, 0.2609, 0.3395]],
#
#         [[0.3309, 0.3786, 0.3913, 0.1165],
#          [0.3750, 0.3643, 0.4130, 0.0326],
#          [0.3971, 0.3929, 0.4130, 0.0281],
#          [0.4265, 0.4071, 0.2609, 0.3395],
#          [0.2500, 0.2286, 0.1812, 0.2194]],
#
#         ...,
#
#         [[0.6471, 0.6500, 0.5942, 0.1509],
#          [0.5735, 0.5643, 0.5000, 0.1859],
#          [0.5147, 0.4857, 0.5217, 0.0422],
#          [0.6324, 0.5929, 0.5362, 0.2346],
#          [0.5147, 0.4857, 0.4710, 0.1288]],
#
#         [[0.5735, 0.5643, 0.5000, 0.1859],
#          [0.5147, 0.4857, 0.5217, 0.0422],
#          [0.6324, 0.5929, 0.5362, 0.2346],
#          [0.5147, 0.4857, 0.4710, 0.1288],
#          [0.4779, 0.5071, 0.5217, 0.2026]],
#
#         [[0.5147, 0.4857, 0.5217, 0.0422],
#          [0.6324, 0.5929, 0.5362, 0.2346],
#          [0.5147, 0.4857, 0.4710, 0.1288],
#          [0.4779, 0.5071, 0.5217, 0.2026],
#          [0.5074, 0.5000, 0.5435, 0.0824]]])

# print('y_train_tensor = ',y_train_tensor)

# y_train_tensor =  tensor([[0.2587],
#         [0.1818],
#         [0.1538],
#         [0.1469],
#
#         [0.5175],
#         [0.4685]])

x_test_tensor = torch.FloatTensor(x_test_data).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test_data).to(DEVICE)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

print('train_dataset = ',train_dataset)
# train_dataset =  <torch.utils.data.dataset.TensorDataset object at 0x00000259FEB2FEE0>

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

################################################################################
#
################################################################################

class MyLSTMModel(nn.Module):
#                          4            4          1
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, data):
        print('data.size(0) =', data.size(0))
        #                        1         배치크기(20)                 4
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)

        outputs, _ = self.lstm(data, (h0, c0))
        last_hs = outputs[:, -1, :]
        prediction = self.fc(last_hs)

        return prediction

model = MyLSTMModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

################################################################################
#
################################################################################

def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    train_loss_sum = 0

    total_train_batch = len(dataloader)

    for inputs, labels in dataloader:
        x_train = inputs.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_avg_loss = train_loss_sum / total_train_batch

    return train_avg_loss

################################################################################
#
################################################################################

def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad():

        val_loss_sum = 0

        total_val_batch = len(dataloader)

        for inputs, labels in dataloader:

            x_val = inputs.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

        val_avg_loss = val_loss_sum / total_val_batch

    return val_avg_loss


################################################################################
#
################################################################################

from datetime import datetime

train_loss_list = []

start_time = datetime.now()

EPOCHS = 500

for epoch in range(EPOCHS):

    avg_loss = model_train(train_loader, model, loss_function, optimizer)

    train_loss_list.append(avg_loss)

    if (epoch % 20 == 0):
        print('epoch: ', epoch, ', train loss = ', avg_loss)

end_time = datetime.now()

print('elapsed time => ', end_time-start_time)

# import matplotlib.pyplot as plt
#
# plt.title('Loss Trend')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.plot(train_loss_list, label='train loss')
# plt.legend()
# plt.show()
#
# test_pred_tensor = model(x_test_tensor)
#
# test_pred_numpy = test_pred_tensor.cpu().detach().numpy()
#
# pred_inverse = scaler_y.inverse_transform(test_pred_numpy)
#
# y_test_numpy = y_test_tensor.cpu().detach().numpy()
#
# y_test_inverse = scaler_y.inverse_transform(y_test_numpy)
#
# import matplotlib.pyplot as plt
#
# plt.plot(y_test_inverse, label='actual')
# plt.plot(pred_inverse, label='prediction')
# plt.grid()
# plt.legend()
#
# plt.show()
#
# import matplotlib.pyplot as plt
#
# plt.plot(y_test_numpy, label='actual')
# plt.plot(test_pred_numpy, label='prediction')
# plt.grid()
# plt.legend()
#
# plt.show()