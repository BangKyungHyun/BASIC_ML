################################################################################
# 라이브러리 호출
################################################################################

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# 데이터셋 가져오기
# https://finance.yahoo.com/quote/subx/history에서 가져옴
################################################################################

data=pd.read_csv('../DATA/SBUX.csv')
print(data.dtypes)
# 스타벅스 주가 데이터넷의 각 컬럼과 데이터 타입을 보여줌
# Date          object
# Open         float64
# High         float64
# Low          float64
# Close        float64
# Adj Close    float64
# Volume         int64
# dtype: object

################################################################################
# 날짜 컬럼을 인덱스로 사용
################################################################################

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

################################################################################
# 데이터 형식 변경
################################################################################

# 데이터 형식을 변경할 때는 astype()을 사용
data['Volume'] = data['Volume'].astype(float)

################################################################################
# 데이터 형식 변경
################################################################################

X=data.iloc[:,:-1]  # 마지막 컬럼을 제외한 모든 컬럼을 x로 사용
y=data.iloc[:,5:6]  # 마지막 Volume을 레이블로 사용
print(X)

#                   Open        High         Low       Close   Adj Close
# Date
# 2023-02-02  110.040001  110.830002  108.000000  109.150002  106.907013
# 2023-02-03  104.580002  106.440002  103.040001  104.300003  102.156685
# 2023-02-06  104.000000  106.169998  103.300003  105.019997  102.861885
# 2023-02-07  104.830002  107.379997  104.559998  106.830002  104.634689
# 2023-02-08  106.320000  106.540001  105.650002  106.300003  104.115585
# ...                ...         ...         ...         ...         ...
# 2024-01-29   93.019997   93.930000   92.239998   93.800003   93.800003
# 2024-01-30   93.000000   94.680000   92.589996   94.080002   94.080002
# 2024-01-31   98.279999   98.360001   93.019997   93.029999   93.029999
# 2024-02-01   93.099998   93.599998   91.870003   93.370003   93.370003
# 2024-02-02   92.690002   93.610001   91.669998   92.989998   92.989998
#
# [252 rows x 5 columns]
print(y)

#                 Volume
# Date
# 2023-02-02   9852900.0
# 2023-02-03  15200500.0
# 2023-02-06   6392200.0
# 2023-02-07   6207500.0
# 2023-02-08   5557500.0
# ...                ...
# 2024-01-29  12728100.0
# 2024-01-30  17625200.0
# 2024-01-31  26751800.0
# 2024-02-01  15001500.0
# 2024-02-02  11365900.0

################################################################################
# 데이터 분포 조정
################################################################################
# 데이터셋에서 데이터 간의 분포가 다르게 나타남
# 분포를 고르게 맞추기 위한 과정이 필요한데 MinMaxScaler()를 사용하여 분산을 조정함
ms = MinMaxScaler()     #  데이터의 모든 값이 0~1사에 존재하도록 분산 조정
ss = StandardScaler()   # 데이터가 평균0, 분산 1이 되도록 분산 조정

X_ss = ss.fit_transform(X)
y_ms = ms.fit_transform(y)

X_train = X_ss[:200, :] # 훈련 데이터셋 (0~199 레코드) 인덱스 처리하여 0이 1번째임
y_train = y_ms[:200, :]

X_test = X_ss[200:, :]  # 테스트 데이터셋 (200~251 레코드)
y_test = y_ms[200:, :]

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

# Training Shape (200, 5) (200, 1)
# Testing Shape (52, 5) (52, 1)

################################################################################
# 데이터셋의 형태 및 크기 조정
################################################################################

# Variable 로 감싸진 텐서는 .backward()가 호출될 때 자동으로 기울기가 계산

X_train_tensors = Variable(torch.Tensor(X_train))
y_train_tensors = Variable(torch.Tensor(y_train))

X_test_tensors = Variable(torch.Tensor(X_test))
y_test_tensors = Variable(torch.Tensor(y_test))

# torch.reshape는 텐서의 형태를 바꿀 때 사용
# 훈련 데이터셋(x_train_tensors)의 형태(200,5)fmf (200,1,5)로 변경하겠다는 의미
# 이와 같이 데이터셋의 형태를 변경하는 이유는 LSTM네트워크의 입력 형태와 맞추기 위함
#                                                           200               ,1, 5 
X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_f.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_f.shape, y_test_tensors.shape)

# Training Shape torch.Size([200, 1, 5]) torch.Size([200, 1])
# Testing Shape torch.Size([52, 1, 5]) torch.Size([52, 1])

################################################################################
# LSTM 네트워크
################################################################################

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes # 클래스 갯수
        self.num_layers = num_layers   # LTSM 계층의 갯수
        self.input_size = input_size   # 입력 크기로 훈련 데이터의 컬럼 갯수(5)를 의미
        self.hidden_size = hidden_size # 은닉층의 뉴런 갯수
        self.seq_length = seq_length   # 시퀀스 길이

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)   # 완전 연결층
        self.relu = nn.ReLU()                     
        self.fc = nn.Linear(128, num_classes)     # 출력층

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # 은닉 상태를 0으로 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # 셀 상태를 0으로 초기화
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # LSTM 계층에 은닉상태와 셀 상태 반영
        hn = hn.view(-1, self.hidden_size)           # 완전연결층 적용을 위해 데이터의 형태 조정(1차원으로 조정)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

################################################################################
# 변수 값 설정
################################################################################

num_epochs = 600000               # 100번의 에포크
learning_rate = 0.0001

input_size = 5                  # 입력 데이터셋의 컬럼(feature) 갯수
hidden_size = 2                 # 은닉층의 뉴런/유닛 갯수
num_layers = 1                  # LSTM 계층의 갯수

num_classes = 1                 # 클래스 갯수
# 앞에서 정의한 값을 이용하여 LSTM 모델 학습
model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

################################################################################
# 모델 학습
################################################################################

for epoch in range(num_epochs):

    outputs = model.forward(X_train_tensors_f) # Feed forward 학습
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors) # 손실 함수를 이용한 오차 계산
    loss.backward()    # 기울기 계산, 역전파
    optimizer.step()   # 오차 업데이트

    if epoch % 10000 == 0:
        print("Epoch: %d, loss: %1.10f" % (epoch, loss.item()))

################################################################################
# 모델 예측 결과를 출력하기 위한 데이터 크기 재구성
################################################################################

df_x_ss = ss.transform(data.iloc[:, :-1])  # 데이터 정규화(분포 조정)
df_y_ms = ms.transform(data.iloc[:, -1:])  # 데이터 정규화

df_x_ss = Variable(torch.Tensor(df_x_ss))
df_y_ms = Variable(torch.Tensor(df_y_ms))
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

################################################################################
# 모델 예측 결과 출력
###############################################################################

train_predict = model(df_x_ss)  # 훈련 데이터셋을 모델에 적용하여 모델 학습
predicted = train_predict.data.numpy() # 모델 학습 결과를 넘파이로 변경
label_y = df_y_ms.data.numpy()

# 모델 학습을 위해 전처리(정규화)했던 것을 해제(그래프의 본래 값을 출력하기 위한 목적)
predicted= ms.inverse_transform(predicted)
label_y = ms.inverse_transform(label_y)
plt.figure(figsize=(10,6))
plt.axvline(x=200, c='r', linestyle='--')

plt.plot(label_y, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.title('Time-Series Prediction')
plt.legend()
plt.show()