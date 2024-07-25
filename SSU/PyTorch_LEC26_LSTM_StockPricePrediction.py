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
# 1.데이터 불러오기
################################################################################
print('1-1.데이터 불러오기 ')
import FinanceDataReader as fdr

# 삼성전자 (005930) 주가
df = fdr.DataReader('005930', '2024-01-01', '2024-07-20')

df = df[ ['Open', 'High', 'Low', 'Volume', 'Close'] ]

print('df.tail(10) =', df.tail(10))

print('len(df) =', len(df))

print('1-2.데이터 불러오기 ')

# len(df) = 136
################################################################################
# 2.데이터 train, test 용 분리하기
################################################################################

# train : test - 70 : 30 분리
print('2-1.데이터 train, test 용 분리하기 ')

SPLIT = int(0.7*len(df))  # train : test = 7 : 3

train_df = df[ :SPLIT ]
test_df = df[ SPLIT: ]

print('2-2.데이터 train, test 용 분리하기 ')

################################################################################
# 3.변수 정규화
################################################################################

print('3-1.변수 정규화 시작 ')

scaler_x = MinMaxScaler()  # feature scaling

train_df.iloc[ : , :-1 ] = scaler_x.fit_transform(train_df.iloc[ : , :-1 ])
test_df.iloc[ : , :-1 ] = scaler_x.fit_transform(test_df.iloc[ : , :-1 ])

scaler_y = MinMaxScaler()  # label scaling

train_df.iloc[ : , -1 ] = scaler_y.fit_transform(train_df.iloc[ : , [-1] ])
test_df.iloc[ : , -1 ] = scaler_y.fit_transform(test_df.iloc[ : , [-1] ])

print('3-2.변수 정규화 끝 ')

################################################################################
# 5. 순차적 Numpy Data 만들기(함수는 실제 호출 시에만 실행됨)
################################################################################

def MakeSeqNumpyData(data, seq_length):
    print('5-1. 순차적 Numpy Data 만들기 start')

    x_seq_list = []
    y_seq_list = []

    # print('len(data) =',len(data))
    # print('data.shape =',data.shape)
    # print('seq_length =',seq_length)

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

    # print('5-2. def MakeSeqNumpyData(data, seq_length)')
    # print('5-2. x_seq_numpy.shape = ',x_seq_numpy.shape)
    # print('5-2. x_seq_numpy.size = ',x_seq_numpy.size)
    # print('5-2. y_seq_numpy.shape = ',y_seq_numpy.shape)
    # print('5-2. y_seq_numpy.size = ',y_seq_numpy.size)

    # def MakeSeqNumpyData(data, seq_length) --> train

    # 5일간의 주가를 바탕으로 6일째 주가를 예측함
    # x_seq_numpy.shape =  (90, 5, 4)
    # x_seq_numpy.size =  1800

    # y_seq_numpy.shape =  (90, 1)
    # y_seq_numpy.size =  90

    # def MakeSeqNumpyData(data, seq_length) --> test
    # x_seq_numpy.shape =  (36, 5, 4)
    # x_seq_numpy.size =  720

    # y_seq_numpy.shape =  (36, 1)
    # y_seq_numpy.size =  36
    print('5-3. 순차적 Numpy Data 만들기 end ')

    return x_seq_numpy, y_seq_numpy

# 1-2.데이터 불러오기
# 2-1.데이터 train, test 용 분리하기
# 2-2.데이터 train, test 용 분리하기
# 3-1.변수 정규화 시작
# 3-2.변수 정규화 끝
# 4-1. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 start
# 5-1. 순차적 Numpy Data 만들기 start
# 5-3. 순차적 Numpy Data 만들기 end
# 5-1. 순차적 Numpy Data 만들기 start
# 5-3. 순차적 Numpy Data 만들기 end
# 4-2. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 end
# 6-1. class MyLSTMModel(nn.Module) 정의 start
# 6-2. class MyLSTMModel(nn.Module) 선언 start
# 6-8. class MyLSTMModel(nn.Module) 정의 end
# 7-1. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
# 6-3. class MyLSTMModel(nn.Module) __init__ start
# 6-4. class MyLSTMModel(nn.Module) __init__ end
# 7-2. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
# 8-1. 메인 처리 start
# 9-1. train 함수 정의 start
# 6-5. class MyLSTMModel(nn.Module) forward start
# 6-6. class MyLSTMModel(nn.Module) forward end
# 6-5. class MyLSTMModel(nn.Module) forward start
# 6-6. class MyLSTMModel(nn.Module) forward end
# 6-5. class MyLSTMModel(nn.Module) forward start
# 6-6. class MyLSTMModel(nn.Module) forward end
# 6-5. class MyLSTMModel(nn.Module) forward start
# 6-6. class MyLSTMModel(nn.Module) forward end
# 6-5. class MyLSTMModel(nn.Module) forward start
# 6-6. class MyLSTMModel(nn.Module) forward end
# 9-2. train 함수 정의 end

################################################################################
# 4. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환
################################################################################

print('4-1. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 start ')

x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEQ_LENGTH)

x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEQ_LENGTH)

# print(x_train_data.shape, y_train_data.shape)
# print(x_test_data.shape, y_test_data.shape)

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

# print('train_dataset = ',train_dataset)
# train_dataset =  <torch.utils.data.dataset.TensorDataset object at 0x00000259FEB2FEE0>

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('4-2. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 end ')

################################################################################
# 6. MyLSTMModel 클래스 정의
# __init__ 부분은 클래스가 호출될 때 1번 실행, forward부분은 실제 실행시마다 호출됨
################################################################################

print('6-1. class MyLSTMModel(nn.Module) 정의 start')

class MyLSTMModel(nn.Module):
    print('6-2. class MyLSTMModel(nn.Module) 선언 start ')
#                          4            4          1
    def __init__(self, input_size, hidden_size, num_layers):

        print('6-3. class MyLSTMModel(nn.Module) __init__ start')
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        print('6-4. class MyLSTMModel(nn.Module) __init__ end ' )

    def forward(self, data):

        # print('6-5. class MyLSTMModel(nn.Module) forward start')

        # print('data.size(0) =', data.size(0))
        # 90개를 20개 단위로 배치 처리함
        # data.size(0) = 20
        # data.size(0) = 20
        # data.size(0) = 20
        # data.size(0) = 20
        # data.size(0) = 10

        # 은닉상태 / 셀상태 초기화
        #                        1         배치크기(20)                 4
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)

        outputs, _ = self.lstm(data, (h0, c0))
        last_hs = outputs[:, -1, :]
        prediction = self.fc(last_hs)

        # print('6-6. class MyLSTMModel(nn.Module) forward end')

        return prediction
        print('6-7. class MyLSTMModel(nn.Module) 선언 end ')

print('6-8. class MyLSTMModel(nn.Module) 정의 end ')

################################################################################
# 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
################################################################################

print('7-1. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')

model = MyLSTMModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('7-2. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')

################################################################################
# 9. train 함수 정의
################################################################################

def model_train(dataloader, model, loss_function, optimizer):

    print('9-1. train 함수 정의 start ')

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

    print('9-2. train 함수 정의 end')

    return train_avg_loss

################################################################################
# 10. 평가 함수 정의
################################################################################

def model_evaluate(dataloader, model, loss_function, optimizer):

    print('10-1. 평가 함수 정의 start')

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

    print('10-2. 평가 함수 정의 end')

    return val_avg_loss

################################################################################
# 8. 메인 처리
################################################################################

print('8-1. 메인 처리 start ')

from datetime import datetime

train_loss_list = []

start_time = datetime.now()

EPOCHS = 500

for epoch in range(EPOCHS):

    avg_loss = model_train(train_loader, model, loss_function, optimizer)

    train_loss_list.append(avg_loss)

    if (epoch % 100 == 0):
        print('epoch: ', epoch, ', train loss = ', avg_loss)

end_time = datetime.now()

print('elapsed time => ', end_time-start_time)

print('8-2. 메인 처리 end')

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