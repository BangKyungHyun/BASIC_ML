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
SEQ_LENGTH = 5          # 정답을 만들기 위해 필요한 시점 개수 time step - 5일간의 주가를 학습하고 6일째 예측함
HIDDEN_SIZE = 4         # RNN 계열 계층을 구성하는 hidden state 개수 (LSTM의 출력 갯수)
NUM_LAYERS = 1          # RNN 계열 계층이 몇 겹으로 쌓였는지 나타냄. 재귀 층(Recurrent Layer)의 개수를 의미
                        # 기본적인 LSTM 모델에서는 Recurrent Layer가 1개 이지만, Layer를 stacking을 통하여 2개의 Layer를 가진 LSTM을 만들 수 있음
LEARNING_RATE = 1e-3    # 학습율
BATCH_SIZE = 20         # 학습을 위한 배치사이즈 개수

################################################################################
# 1.데이터 불러오기
################################################################################
# print('1-1.데이터 불러오기 ')
import FinanceDataReader as fdr

# 삼성전자 (005930) 주가, 시작일자, 종료일자
df = fdr.DataReader('005930', '2020-01-02', '2024-07-20')
# df = fdr.DataReader('005930', '2024-06-02', '2024-07-20')

df = df[ ['Open', 'High', 'Low', 'Volume', 'Close'] ]

print('df.head(10) =', df.head(10))

# df.tail(10) =
# Date        Open   High    Low    Volume  Close
# 2024-07-08  87900  88600  86900  24035809  87400
# 2024-07-09  87800  88200  86900  21336201  87800
# 2024-07-10  87600  88000  87100  17813847  87800
# 2024-07-11  88500  88800  86700  24677608  87600
# 2024-07-12  85900  86100  84100  26344386  84400
# 2024-07-15  84700  87300  84100  25193080  86700
# 2024-07-16  86900  88000  86700  16166688  87700
# 2024-07-17  87100  88000  86400  18186490  86700
# 2024-07-18  83800  86900  83800  24721790  86900
# 2024-07-19  85600  86100  84100  18569122  84400

# print('len(df) =', len(df))
# len(df) = 1123

# print('df.shape =', df.shape)
# df.shape = (1123, 5)

# print('1-2.데이터 불러오기 ')

################################################################################
# 2.데이터 train, test 용 분리하기
################################################################################

# train : test - 70 : 30 분리
# print('2-1.데이터 train, test 용 분리하기 ')

SPLIT = int(0.7*len(df))  # train : test = 7 : 3

train_df = df[ :SPLIT ]
test_df = df[ SPLIT: ]
#
# print('train_df.shape =', train_df.shape)
# print('test_df.shape =', test_df.shape)
# #
# train_df.shape = (786, 5)
# test_df.shape = (337, 5)

# print('2-2.데이터 train, test 용 분리하기 ')

################################################################################
# 3.데이터 정규화  feature를 리스케일링 하여 feature의 평균이(mean) 0 분산이(variance) 1이 되게 만들어 주는 과정
################################################################################

# print('3-1.데이터 정규화 시작 ')

# data.shape[0] 하면 data라는 이름의 data frame의 전체 row수 출력 가능

scaler_x = MinMaxScaler()  # feature scaling  -  Open   High    Low    Volume 4개 정규화

train_df.iloc[ : , :-1 ] = scaler_x.fit_transform(train_df.iloc[ : , :-1 ])
test_df.iloc[ : , :-1 ] = scaler_x.transform(test_df.iloc[ : , :-1 ])
# test_df.iloc[ : , :-1 ] = scaler_x.fit_transform(test_df.iloc[ : , :-1 ])

# print('train_df.iloc[ : , :-1 ] =', train_df.iloc[ : , :-1 ])
# train_df.iloc[ : , :-1 ] =
# Date        Open        High      Low       Volume
# 2024-01-02  0.485294  0.557143  0.543478  0.154374
# 2024-01-03  0.507353  0.485714  0.456522  0.250531
# 2024-01-04  0.330882  0.378571  0.391304  0.116451
# 2024-01-05  0.375000  0.364286  0.413043  0.032613
# 2024-01-08  0.397059  0.392857  0.413043  0.028117
# ...              ...       ...       ...       ...
# 2024-05-16  0.632353  0.592857  0.536232  0.234601
# 2024-05-17  0.514706  0.485714  0.471014  0.128834
# 2024-05-20  0.477941  0.507143  0.521739  0.202630
# 2024-05-21  0.507353  0.500000  0.543478  0.082378
# 2024-05-22  0.477941  0.478571  0.478261  0.203980
#
# [95 rows x 4 columns]

# print('test_df.iloc[ : , :-1 ] =\n', test_df.iloc[ : , :-1 ])
# test_df.iloc[ : , :-1 ] =
# Date        Open      High       Low    Volume
# 2024-05-23  0.234043  0.312057  0.264706  0.255183
# 2024-05-24  0.170213  0.163121  0.161765  0.507368
# 2024-05-27  0.063830  0.248227  0.036765  0.945893
# 2024-05-28  0.148936  0.234043  0.198529  0.249981
# 2024-05-29  0.234043  0.248227  0.125000  0.572142

scaler_y = MinMaxScaler()  # label scaling - close 1개 정규화

train_df.iloc[ : , -1 ] = scaler_y.fit_transform(train_df.iloc[ : , [-1] ])
test_df.iloc[ : , -1 ] = scaler_y.transform(test_df.iloc[ : , [-1] ])
# test_df.iloc[ : , -1 ] = scaler_y.fit_transform(test_df.iloc[ : , [-1] ])

# print('train_df.iloc[ : , -1 ] = \n', train_df.iloc[ : , -1 ])

# train_df.iloc[ : , -1 ] =
#  Date         Close
# 2024-01-02    0.601399
# 2024-01-03    0.419580
# 2024-01-04    0.391608
# 2024-01-05    0.391608
# 2024-01-08    0.384615
#                 ...
# 2024-05-16    0.503497
# 2024-05-17    0.447552
# 2024-05-20    0.552448
# 2024-05-21    0.517483
# 2024-05-22    0.468531
# Length: 95, dtype: float64

# print('test_df.iloc[ : , -1 ] = \n', test_df.iloc[ : , -1 ])
# test_df.iloc[ : , -1 ] =
#  Date
# 2024-05-23    0.335664
# 2024-05-24    0.167832

# print('3-2.데이터 정규화 끝 ')

################################################################################
# 5. 순차적 Numpy Data 만들기(함수는 실제 호출 시에만 실행됨)
# 전체 기간(시작일자~종료일자) 동안 일자별로 5영업일 동안의 주가 흐름을 바탕으로 6일자의 종가를 예측함
# X data : 781일 동안 일별로 5영업일 간의 4개 피쳐(Open, High,Low,Volume) 데이터
# Y data : 781일 동안 일별로 6영업일의 1개 피쳐(Close) 데이터
################################################################################

def MakeSeqNumpyData(data, seq_length):
    # print('5-1. 순차적 Numpy Data 만들기 start')

    # print('data.head(10) =', data)
    # data.head(10) =
    # [[0.         0.04597701 0.         0.18168904 0.05319149]  index 0 1일
    #  [0.05882353 0.01149425 0.08536585 0.13494037 0.0106383 ]  index 1 2일
    #  [0.42352941 0.22988506 0.31707317 0.39888141 0.23404255]  index 2 3일
    #  [0.47058824 0.29885057 0.35365854 0.3410749  0.22340426]  index 3 4일
    #  [0.2        0.06896552 0.17073171 0.14949511 0.05319149]  index 4 5일
    #                                               ---------  6일째 종가는 0.
    #  [0.17647059 0.         0.1097561  0.2184582  0.        ]  index 5 6일
    #                                               ---------

    x_seq_list = []
    y_seq_list = []

    # print('len(data) =',len(data))
    # print('data.shape =',data.shape)
    # print('seq_length =',seq_length)

    # len(data) = 786  --> train
    # data.shape = (786, 5)    Open   High    Low    Volume  Close
    # seq_length = 5

    # len(data) = 337  --> test
    # data.shape = (337, 5)
    # seq_length = 5

    for i in range(len(data) - seq_length):

        x_seq_list.append(data[i:i + seq_length, :-1])   # 5일간의 Open   High    Low    Volume 데이터
        y_seq_list.append(data[i + seq_length, [-1]])    # 5일째 Close 주가
        # print('i:i + seq_length =', i,i + seq_length)
        # i: i + seq_length = 0  5

        # print('i + seq_length =', i + seq_length)
        # i + seq_length = 5

        # data[i:i + seq_length, :-1] =
        # [[0.         0.04597701 0.         0.18168904]
        #  [0.05882353 0.01149425 0.08536585 0.13494037]
        #  [0.42352941 0.22988506 0.31707317 0.39888141]
        #  [0.47058824 0.29885057 0.35365854 0.3410749 ]
        #  [0.2        0.06896552 0.17073171 0.14949511]]

        # print('data[i + seq_length, [-1]]) =', data[i + seq_length, [-1]])
        # data[i + seq_length, [-1]]) = [0.]  index 5 (6일째 종가)

    x_seq_numpy = np.array(x_seq_list)
    y_seq_numpy = np.array(y_seq_list)

    # print('5-2. def MakeSeqNumpyData(data, seq_length)')
    # print('5-2. x_seq_numpy.shape = ',x_seq_numpy.shape)
    # print('5-2. x_seq_numpy.size = ',x_seq_numpy.size)
    # print('5-2. y_seq_numpy.shape = ',y_seq_numpy.shape)
    # print('5-2. y_seq_numpy.size = ',y_seq_numpy.size)

    # 5일간의 주가를 바탕으로 6일째 주가를 예측함
    # 5-2. def MakeSeqNumpyData(data, seq_length) --> train
    #                           781일 동안 일별로 5영업일 간의 4개 피쳐(Open, High,Low,Volume) 데이터
    # 5-2. x_seq_numpy.shape =  (781, 5, 4)
    # 5-2. x_seq_numpy.size =  15620
    #                           781일 동안 일별로 6영업일의 1개 피쳐(Close) 데이터
    # 5-2. y_seq_numpy.shape =  (781, 1)
    # 5-2. y_seq_numpy.size =  781

    # 5-2. def MakeSeqNumpyData(data, seq_length) --> test

    # 5-2. x_seq_numpy.shape =  (332, 5, 4)
    # 5-2. x_seq_numpy.size =  6640

    # 5-2. y_seq_numpy.shape =  (332, 1)
    # 5-2. y_seq_numpy.size =  332

    # print('5-3. 순차적 Numpy Data 만들기 end ')

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
# 시퀀스 길이가 5, 즉 time step 갯수가 5개인 numpy 타입의 train, test data 생성
################################################################################

# print('4-1. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 start ')

x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEQ_LENGTH)
x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEQ_LENGTH)

# print(x_train_data.shape, y_train_data.shape)
# print(x_test_data.shape, y_test_data.shape)

# (90, 5, 4) (90, 1)  -> train
# (36, 5, 4) (36, 1)  -> test

# numpy 데이터로부터 텐서를 생성 (train 데이터)
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

# numpy 데이터로부터 tensor 생성 (test 데이터)
x_test_tensor = torch.FloatTensor(x_test_data).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test_data).to(DEVICE)

# tensor 데이터로 부터 train/test dataset 생성
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# print('train_dataset = ',train_dataset)
# train_dataset =  <torch.utils.data.dataset.TensorDataset object at 0x00000259FEB2FEE0>

# 배치크기가 20인 DataLoader 생성
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print('4-2. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 end ')

################################################################################
# 6. MyLSTMModel 클래스 정의
# __init__ 부분은 클래스가 호출될 때 1번 실행, forward부분은 실제 실행 시 마다 호출됨
################################################################################

# print('6-1. class MyLSTMModel(nn.Module) 정의 start')

# input_size: input x에 대한 features의 수
# hidden_size: hidden state의 features의 수
# num_layers: LSTM을 스택킹(stacking)하는 수
# batch_first: 입출력 텐서의 형태가 다음과 같음. 기본값은 False
# True로 설정시 (batch, seq, feature)
# False로 설정시 (seq, batch, feature)
# (주의사항) hidden_state, cell_state에는 영향을 미치지 않습니다.
# bidirectional: 양방향 LSTM 구현 여부. 기본값은 False

class MyLSTMModel(nn.Module):
    # print('6-2. class MyLSTMModel(nn.Module) 선언 start ')
#                          4            4          1
    def __init__(self, input_size, hidden_size, num_layers):

        # print('6-3. class MyLSTMModel(nn.Module) __init__ start')
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=False, batch_first=True)

        # print('self.lstm =',self.lstm)
        # self.lstm = LSTM(4, 4, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        # print('6-4. class MyLSTMModel(nn.Module) __init__ end ' )

    def forward(self, data):

        ############################################################
        #     h1         h2         h3         h4       last_hs
        #      ^          ^          ^          ^          ^
        #      |---|      |---|      |---|      |---|      |
        #    OOOO  |    OOOO  |    OOOO  |    OOOO  |    OOOO (hidden size : 4 )
        #      |   |      |   |      |   |      |   |      |
        #    lstm  |    lstm  |    lstm  |    lstm  |    lstm
        #      ^   |      ^   |      ^   |      ^   |      ^
        #      |   |      |   |      |   |      |   |      |
        #      +   |----> +   |----> +   |----> +   |----> +
        #      |          |          |          |          |
        #    input      input      input      input      input
        #      ^          ^          ^          ^          ^
        #      |          |          |          |          |
        #  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]
        #  [O,H,L,V] = [Open, High, Low, Volume]
        ############################################################

        # print('6-5. class MyLSTMModel(nn.Module) forward start')

        # print('data.shape =', data.shape)
        # print('data =', data)
        #
        # data.shape = torch.Size([20, 5, 4]) => batch_size(20) sequence_length(5)  input_size(4)

        # data = tensor([[[0.4853, 0.5571, 0.5435, 0.1544],
        #          [0.5074, 0.4857, 0.4565, 0.2505],
        #          [0.3309, 0.3786, 0.3913, 0.1165],
        #          [0.3750, 0.3643, 0.4130, 0.0326],
        #          [0.3971, 0.3929, 0.4130, 0.0281]],
        #           20 개 반복

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

        # 리턴값 outputs은 모든 시점(time step)의 hidden state들이며 last_hs는 마지막 시점(time step)의 hidden state를 나타냄
        # 즉 outputs 는 [h1, h2, h3, h4, last_hs]
        outputs, (hidden_state, cell_state) = self.lstm(data, (h0, c0))

        # print('outputs.shape, hidden_state.shape, cell_state.shape =',outputs.shape, hidden_state.shape, cell_state.shape)

        # outputs.shape = torch.Size([20, 5, 4])      batch_size, sequence_length, input_size
        # hidden_state.shape = torch.Size([1, 20, 4]) num_layers, batch_size, hidden_size
        # cell_state.shape = torch.Size([1, 20, 4])   num_layers, batch_size, hidden_size

        # 개별 outputs 에서 sequence가 마지막(5번째)인 경우에만 가져옴
        last_hs = outputs[:, -1, :]

        # print('outputs[:, :, :] =', outputs[:, :, :])
        # print('last_hs =', last_hs)

        # outputs[:, :, :] =
        # tensor([[[ 0.0514, -0.2357,  0.1682,  0.1654],
        #          [ 0.0959, -0.3589,  0.2979,  0.3061],
        #          [ 0.1289, -0.4321,  0.3908,  0.4114],
        #          [ 0.1530, -0.4813,  0.4583,  0.4853],
        #          [ 0.1701, -0.5109,  0.5079,  0.5380]],   <--5번째 마지막 데이터
        #                    ...........
        #
        #         [[ 0.0499, -0.2392,  0.1768,  0.1680],
        #          [ 0.0921, -0.3580,  0.3138,  0.3131],
        #          [ 0.1229, -0.4295,  0.4154,  0.4236],
        #          [ 0.1481, -0.4872,  0.4876,  0.4982],
        #          [ 0.1692, -0.5265,  0.5343,  0.5471]],  <--5번째 마지막 데이터
        #
        #         [[ 0.0500, -0.2359,  0.1733,  0.1665],
        #          [ 0.0903, -0.3516,  0.3155,  0.3127],
        #          [ 0.1231, -0.4330,  0.4166,  0.4213],
        #          [ 0.1496, -0.4883,  0.4837,  0.4951],
        #          [ 0.1698, -0.5217,  0.5271,  0.5458]]],  <--5번째 마지막 데이터
        #           grad_fn=<SliceBackward0>)
 
        # outputs 중에 time step(sequence 길이) 가 5번째 데이터만 모음
        # last_hs =
        # tensor([[ 0.1701, -0.5109,  0.5079,  0.5380],
        #                          .............
        #         [ 0.1692, -0.5265,  0.5343,  0.5471],
        #         [ 0.1698, -0.5217,  0.5271,  0.5458]],
        #         grad_fn=<SliceBackward0>)

        prediction = self.fc(last_hs)

        # print('prediction =\n',prediction )

        # prediction =
        # tensor([[0.4161],
        #         [0.4101],
        #         [0.4020],
        #         [0.4027],
        #         [0.4033],
        #         [0.4006],
        #         [0.4017],
        #         [0.4083],
        #         [0.4141],
        #         [0.4117],
        #         [0.4098],
        #         [0.4080],
        #         [0.4033],
        #         [0.3934],
        #         [0.3898],
        #         [0.3893],
        #         [0.3783],
        #         [0.3729],
        #         [0.3714],
        #         [0.3682]], grad_fn= < AddmmBackward0 >)

        # print('6-6. class MyLSTMModel(nn.Module) forward end')

        return prediction
        # print('6-7. class MyLSTMModel(nn.Module) 선언 end ')

# print('6-8. class MyLSTMModel(nn.Module) 정의 end ')

################################################################################
# 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
################################################################################

# print('7-1. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')

model = MyLSTMModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print('7-2. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')

################################################################################
# 9. train 함수 정의
################################################################################

def model_train(dataloader, model, loss_function, optimizer):

    # print('9-1. train 함수 정의 start ')

    model.train()

    train_loss_sum = 0

    total_train_batch = len(dataloader)

    for inputs, labels in dataloader:   # inputs : Open, High,Low,Volume
        x_train = inputs.to(DEVICE)     # labels : Close
        y_train = labels.to(DEVICE)

        outputs = model(x_train)        # 입력 데이터에 대한 예측값 계산(6번째 종가)
        loss = loss_function(outputs, y_train) # 모델 예측값과 정답간의 오차(loss)인 손실함수 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_avg_loss = train_loss_sum / total_train_batch # 학습 데이터 평균 오차 계산

    # print('9-2. train 함수 정의 end')

    return train_avg_loss

################################################################################
# 10. 평가 함수 정의
################################################################################

def model_evaluate(dataloader, model, loss_function, optimizer):

    # print('10-1. 평가 함수 정의 start')

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

    # print('10-2. 평가 함수 정의 end')

    return val_avg_loss

################################################################################
# 8. 메인 처리
################################################################################

# print('8-1. 메인 처리 start ')

from datetime import datetime

train_loss_list = []

start_time = datetime.now()

EPOCHS = 50000

for epoch in range(EPOCHS):

    avg_loss = model_train(train_loader, model, loss_function, optimizer)

    train_loss_list.append(avg_loss)

    if (epoch % 10000 == 0):
        print('epoch: ', epoch, ', train loss = ', avg_loss)

end_time = datetime.now()

print('elapsed time => ', end_time-start_time)

print('8-2. 메인 처리 end')

################################################################################
# train data 학습 결과
################################################################################

import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss_list, label='train loss')
plt.legend()
plt.show()

################################################################################
# test data 예측과 실제 데이터 비교 (스케일링 되기 전의 원래값으로 변환 후)
################################################################################

test_pred_tensor = model(x_test_tensor)  # 328 개 test 데이터에 대한 예측 수행

test_pred_numpy = test_pred_tensor.cpu().detach().numpy()  # 예측값을 numpy로 변환

pred_inverse = scaler_y.inverse_transform(test_pred_numpy) # 스케일링 되기 전의 원래값으로 변환

y_test_numpy = y_test_tensor.cpu().detach().numpy()        # 정답을 numpy로 변환

y_test_inverse = scaler_y.inverse_transform(y_test_numpy)  # 스케일링 되기 전의 원래값으로 변환

import matplotlib.pyplot as plt

plt.plot(y_test_inverse, label='actual')
plt.plot(pred_inverse, label='prediction')
plt.grid()
plt.legend()

plt.show()

################################################################################
# test data 예측과 실제 데이터 비교 (스케일링 되기 전의 원래값으로 변환 전)
################################################################################
import matplotlib.pyplot as plt

plt.plot(y_test_numpy, label='actual')
plt.plot(test_pred_numpy, label='prediction')
plt.grid()
plt.legend()

plt.show()