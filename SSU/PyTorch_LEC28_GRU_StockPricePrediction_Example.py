import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")

FEATURE_NUMS = 4        # 입력층으로 들어가는 데이터 개수 feature
SEQ_LENGTH = 5          # 정답을 만들기 위해 필요한 시점 개수 time step (6일째 주가를 예측하기 위한 5일간의 데이터)
HIDDEN_SIZE = 4         # RNN 계열 계층을 구성하는 hidden state 개수 (GRU가 만드는 출력값)
NUM_LAYERS = 1          # RNN 계열 계층이 몇겹으로 쌓였는지 나타냄
LEARNING_RATE = 1e-3    # 학습율
BATCH_SIZE = 20         # 학습을 위한 배치사이즈 개수

# pip install finance-datareader

################################################################################
# 1.데이터 불러오기
################################################################################

# print('1-1.데이터 불러오기 ')

import FinanceDataReader as fdr

df = fdr.DataReader('005930', '2020-01-01', '2024-06-30')

df = df[ ['Open', 'High', 'Low', 'Volume', 'Close'] ]

# print('df.head(10) =', df.head(10))

# df.head(10) =
# Date        Open   High    Low    Volume  Close
# 2020-01-02  55500  56000  55000  12993228  55200
# 2020-01-03  56000  56600  54900  15422255  55500
# 2020-01-06  54900  55600  54600  10278951  55500
# 2020-01-07  55700  56400  55600  10009778  55800
# 2020-01-08  56200  57400  55900  23501171  56800
# 2020-01-09  58400  58600  57400  24102579  58600
# 2020-01-10  58800  59700  58300  16000170  59500
# 2020-01-13  59600  60000  59100  11359139  60000
# 2020-01-14  60400  61000  59900  16906295  60000
# 2020-01-15  59500  59600  58900  14300928  59000

# print('len(df) =', len(df))
# len(df) = 1108

# print('df.shape =', df.shape)
# df.shape = (1108, 5)

df['Close'].plot().grid()

################################################################################
# 2.데이터 train, test 용 분리하기
################################################################################
# print('1-2.데이터 불러오기 ')

# train : test - 70 : 30 분리
# print('2-1.데이터 train, test 용 분리하기 ')

SPLIT = int(0.7*len(df))  # train : test = 7 : 3

train_df = df[ :SPLIT ]
test_df = df[ SPLIT: ]

# print('2-2.데이터 train, test 용 분리하기 ')

################################################################################
# 3.데이터 정규화  feature를 리스케일링 하여 feature의 평균이(mean) 0 분산이(variance) 1이 되게 만들어 주는 과정
################################################################################

# print('3-1.데이터 정규화 시작 ')

# data.shape[0] 하면 data라는 이름의 data frame의 전체 row수 출력 가능

# MinMaxScaler() 이용해서 0~1 값을 가지도록 feature scaling과 label scaling 각각 수행함


# 입력값 추출
scaler_x = MinMaxScaler()  # feature scaling

train_df.iloc[ : , :-1 ] = scaler_x.fit_transform(train_df.iloc[ : , :-1 ])
test_df.iloc[ : , :-1 ] = scaler_x.fit_transform(test_df.iloc[ : , :-1 ])

# print('train_df.iloc[ : , :-1 ] =', train_df.iloc[ : , :-1 ])
#
# train_df.iloc[ : , :-1 ] =
# Date        Open      High       Low    Volume
# 2020-01-02  0.270440  0.233803  0.269068  0.085468
# 2020-01-03  0.280922  0.245070  0.266949  0.114201
# 2020-01-06  0.257862  0.226291  0.260593  0.053361
# 2020-01-07  0.274633  0.241315  0.281780  0.050177
# 2020-01-08  0.285115  0.260094  0.288136  0.209766
# ...              ...       ...       ...       ...
# 2023-02-13  0.425577  0.365258  0.423729  0.058701
# 2023-02-14  0.440252  0.382160  0.442797  0.039731
# 2023-02-15  0.446541  0.382160  0.417373  0.088010
# 2023-02-16  0.417191  0.378404  0.425847  0.094998
# 2023-02-17  0.425577  0.370892  0.425847  0.059421
#
# [775 rows x 4 columns]

# print('test_df.iloc[ : , :-1 ] =\n', test_df.iloc[ : , :-1 ])
#
# test_df.iloc[ : , :-1 ] =
#
# Date        Open      High       Low    Volume
# 2023-02-20  0.142308  0.132075  0.109804  0.136570
# 2023-02-21  0.134615  0.124528  0.117647  0.035484
# 2023-02-22  0.088462  0.086792  0.078431  0.118274
# 2023-02-23  0.096154  0.113208  0.098039  0.139251
# 2023-02-24  0.119231  0.116981  0.090196  0.086443
# ...              ...       ...       ...       ...
# 2024-06-24  0.788462  0.807547  0.803922  0.185661
# 2024-06-25  0.823077  0.841509  0.827451  0.255730
# 2024-06-26  0.803846  0.826415  0.819608  0.230565
# 2024-06-27  0.850000  0.833962  0.843137  0.114044
# 2024-06-28  0.873077  0.845283  0.854902  0.070012
#
# [333 rows x 4 columns]

# 여기서 lalel saling을 별도로 하는 이유는 우리가 test data로 예측을 수행하면 0~1 범위를
# 가지는 예측값이 나오기 때문에 scaling 되기 전의 이러한 0~1에 대응되는 원래의 값(label)으로
# 바뀌기 위해서 scaler의 inverse_transform 함수를 사용하기 위한 목적임

# Label 추출

scaler_y = MinMaxScaler()  # label scaling

train_df.iloc[ : , -1 ] = scaler_y.fit_transform(train_df.iloc[ : , [-1] ])
test_df.iloc[ : , -1 ] = scaler_y.fit_transform(test_df.iloc[ : , [-1] ])

# print('train_df.iloc[ : , -1 ] = \n', train_df.iloc[ : , -1 ])

# train_df.iloc[ : , -1 ] =
#  Date          Close
# 2020-01-02    0.261856
# 2020-01-03    0.268041
# 2020-01-06    0.268041
# 2020-01-07    0.274227
# 2020-01-08    0.294845
#                 ...
# 2023-02-13    0.420619
# 2023-02-14    0.426804
# 2023-02-15    0.406186
# 2023-02-16    0.437113
# 2023-02-17    0.414433
# , Length: 775, dtype: float64

# print('test_df.iloc[ : , -1 ] = \n', test_df.iloc[ : , -1 ])
#
# test_df.iloc[ : , -1 ] =
#  Date         Name: Close,
# 2023-02-20    0.140684
# 2023-02-21    0.117871
# 2023-02-22    0.079848
# 2023-02-23    0.114068
# 2023-02-24    0.087452
#                 ...
# 2024-06-24    0.821293
# 2024-06-25    0.828897
# 2024-06-26    0.847909
# 2024-06-27    0.859316
# 2024-06-28    0.855513
#  Length: 333, dtype: float64

# print('3-2.데이터 정규화 끝 ')

################################################################################
# 5. 순차적 Numpy Data 만들기(함수는 실제 호출 시에만 실행됨)
# 전체 기간(시작일자~종료일자) 동안 개별 일자별로 5영업일 동안의 주가 흐름을 바탕으로 6일자의 종가를 예측함
# X data : 1108일 동안 일별로 5영업일 간의 4개 피쳐(Open, High,Low,Volume) 데이터
# Y data : 1108일 동안 일별로 6영업일의 1개 피쳐(Close) 데이터
################################################################################

# print('5-1. 순차적 Numpy Data 만들기 start')

def MakeSeqNumpyData(data, seq_length):

    # print('5-2. 순차적 Numpy Data 만들기 def start')

    # print('data.head(10) =', data)

    # print('data.head(10) =', data)
    # data.head(10) =
    # [[0.         0.04597701 0.         0.18168904 0.05319149]  index 0 1일
    #  [0.05882353 0.01149425 0.08536585 0.13494037 0.0106383 ]  index 1 2일
    #  [0.42352941 0.22988506 0.31707317 0.39888141 0.23404255]  index 2 3일
    #  [0.47058824 0.29885057 0.35365854 0.3410749  0.22340426]  index 3 4일
    #  [0.2        0.06896552 0.17073171 0.14949511 0.05319149]  index 4 5일
    #                                               ---------  6일째 종가는 0.
    #  [0.17647059 0.         0.1097561  0.2184582  0.        ]<=index 5 6일
    #                                               ---------

    x_seq_list = []
    y_seq_list = []

    # print('len(data) =',len(data))
    # print('data.shape =',data.shape)
    # print('seq_length =',seq_length)

    ########################
    # train data
    ########################
    # len(data) = 775
    # data.shape = (775, 5)
    # seq_length = 5

    ########################
    # test data
    ########################
    # len(data) = 333
    # data.shape = (333, 5)
    # seq_length = 5

    # train data는 전체가 775일이므로 770일 까지의 입력 데이터을 가지고 775일째 종가를 예측할 수 있음 (770 for문 수행)
    # test  data는 전체가 333일이므로 328일 까지의 입력 데이터을 가지고 333일째 종가를 예측할 수 있음 (328 for문 수행)

    # 775          5
    for i in range(len(data) - seq_length):

        x_seq_list.append(data[ i:i+seq_length, :-1 ])
        y_seq_list.append(data[ i+seq_length, [-1] ])
        # print('i =', i)

    # 넘파이의 array 함수에 리스트를 넣으면 ndarray 클래스 객체 즉, 배열로 변환해 줌
    # 많은 숫자 데이터를 하나의 변수에 넣고 관리 할 때 리스트는 속도가 느리고 메모리를 많이 차지하는 단점이 있음
    # 배열(array)을 사용하면 적은 메모리로 많은 데이터를 빠르게 처리할 수 있음

    x_seq_numpy = np.array(x_seq_list)
    y_seq_numpy = np.array(y_seq_list)

    # print('5-3. 순차적 Numpy Data 만들기 def end')

    # print('x_seq_numpy =', x_seq_numpy)
    # print('y_seq_numpy =', y_seq_numpy)

    # np.array() 처리를 한번 더해서 대괄호[]가 한번 더 생김
    # x_seq_numpy =
    # [[[0.27044025 0.23380282 0.2690678  0.0854681 ]
    #   [0.28092243 0.24507042 0.26694915 0.11420097]
    #   [0.25786164 0.22629108 0.26059322 0.05336102]
    #   [0.27463312 0.24131455 0.28177966 0.05017699]
    #   [0.2851153  0.2600939  0.28813559 0.20976616]]

    return x_seq_numpy, y_seq_numpy

# print('5-4. 순차적 Numpy Data 만들기 end ')

################################################################################
# 4. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환
# 시퀀스 길이가 5, 즉 time step 갯수가 5개인 numpy 타입의 train, test data 생성
################################################################################

# print('4-1. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 start ')

# print('train_df =', train_df)
# print('np.array(train_df) =', np.array(train_df))
#
# train_df =                 Open      High       Low    Volume     Close
# Date
# 2020-01-02  0.270440  0.233803  0.269068  0.085468  0.261856
# 2020-01-03  0.280922  0.245070  0.266949  0.114201  0.268041
# 2020-01-06  0.257862  0.226291  0.260593  0.053361  0.268041
# 2020-01-07  0.274633  0.241315  0.281780  0.050177  0.274227
# 2020-01-08  0.285115  0.260094  0.288136  0.209766  0.294845
# ...              ...       ...       ...       ...       ...
# 2023-02-13  0.425577  0.365258  0.423729  0.058701  0.420619
# 2023-02-14  0.440252  0.382160  0.442797  0.039731  0.426804
# 2023-02-15  0.446541  0.382160  0.417373  0.088010  0.406186
# 2023-02-16  0.417191  0.378404  0.425847  0.094998  0.437113
# 2023-02-17  0.425577  0.370892  0.425847  0.059421  0.414433
#
# [775 rows x 5 columns]
# np.array(train_df) =
# [[0.27044025 0.23380282 0.2690678  0.0854681  0.26185567]
#  [0.28092243 0.24507042 0.26694915 0.11420097 0.26804124]
#  [0.25786164 0.22629108 0.26059322 0.05336102 0.26804124]
#  ...
#  [0.44654088 0.38215962 0.41737288 0.08800985 0.40618557]
#  [0.41719078 0.37840376 0.42584746 0.09499755 0.4371134 ]
#  [0.42557652 0.37089202 0.42584746 0.05942117 0.41443299]]

x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEQ_LENGTH)
x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEQ_LENGTH)

# print(x_train_data.shape, y_train_data.shape)
# print(x_test_data.shape, y_test_data.shape)
#
# (770, 5, 4) (770, 1) => 4개의 입력(open, high, low, volume) 데이터가 5일 단위로 770개(일)가 생성
# (328, 5, 4) (328, 1)

# numpy 데이터로부터 tensor 데이터 생성
x_train_tensor = torch.FloatTensor(x_train_data).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train_data).to(DEVICE)
#
# print('x_train_tensor = ',x_train_tensor)
# print('y_train_tensor = ',y_train_tensor)

# x_train_tensor =  \
# tensor([[[0.2704, 0.2338, 0.2691, 0.0855],
#          [0.2809, 0.2451, 0.2669, 0.1142],
#          [0.2579, 0.2263, 0.2606, 0.0534],
#          [0.2746, 0.2413, 0.2818, 0.0502],
#          [0.2851, 0.2601, 0.2881, 0.2098]],
#
#
#         [[0.4193, 0.3653, 0.4258, 0.0428],
#          [0.4256, 0.3653, 0.4237, 0.0587],
#          [0.4403, 0.3822, 0.4428, 0.0397],
#          [0.4465, 0.3822, 0.4174, 0.0880],
#          [0.4172, 0.3784, 0.4258, 0.0950]]])

x_test_tensor = torch.FloatTensor(x_test_data).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test_data).to(DEVICE)

# tensor데이터로부터 train dataset / test dataset 생성
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# print('train_dataset = ',train_dataset)
# train_dataset =  <torch.utils.data.dataset.TensorDataset object at 0x00000259FEB2FEE0>

# 배치 사이즈 20인 데이터로더 생성
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print('4-2. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 end ')

################################################################################
# 6. MyGRUModel 클래스 정의
# __init__ 부분은 객체가 생성될 때 1번 실행, forward 부분은 실제 실행 시 마다 호출됨
################################################################################

# print('6-1. class MyGRUModel(nn.Module) 정의 start')

class MyGRUModel(nn.Module):

    # print('6-2. class MyGRUModel(nn.Module) 선언 start ')

    # input_size: input x에 대한 features의 수 (open, high, low, volume)
    # hidden_size: hidden state의 features의 수 (hyper parameter)
    # num_layers: GRU를 스택킹(stacking)하는 수 (hyper parameter)
    # batch_first: 입출력 텐서의 형태가 다음과 같음. 기본값은 False = True로 설정시 (batch, seq, feature) False로 설정시 (seq, batch, feature)
    # (주의사항) hidden_state에는 영향을 미치지 않습니다.
    #                        4           4           1
    # model = MyGRUModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    def __init__(self, input_size, hidden_size, num_layers):

        # print('6-3. class MyGRUModel(nn.Module) __init__ start')

        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #                       4
        self.fc = nn.Linear(hidden_size, 1)

        # print('self.gru =',self.gru)
        # print('6-4. class MyGRUModel(nn.Module) __init__ end ' )

    def forward(self, data):

        # print('6-5. class MyGRUModel(nn.Module) forward start')

        ############################################################
        #     1일        2일        3일         4일       5일
        #     h1         h2         h3         h4       last_hs
        ############################################################
        #      ^          ^          ^          ^          ^
        #      |---|      |---|      |---|      |---|      |
        #    OOOO  |    OOOO  |    OOOO  |    OOOO  |    OOOO (hidden size : 4 )
        #      |   |      |   |      |   |      |   |      |
        #     GRU  |     GRU  |     GRU  |     GRU  |     GRU
        #      ^   |      ^   |      ^   |      ^   |      ^
        #      |   |      |   |      |   |      |   |      |
        #      +   |----> +   |----> +   |----> +   |----> +
        #      |          |          |          |          |
        #    input      input      input      input      input (입력값 :4 )
        #      ^          ^          ^          ^          ^
        #      |          |          |          |          |
        #  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]  [O,H,L,V]
        #  [O,H,L,V] = [Open, High, Low, Volume]
        ############################################################

        # print('data.shape =', data.shape)
        # print('data =', data)
        # print('data.size(0) =', data.size(0))

        # data.shape = torch.Size([20, 5, 4]) => batch_size(20) sequence_length(5)  input_size(4)
        # 4개의 feature(Open, High, Low, Volume) 가 5일 단위로 20개 쌍이 있음
        # data =
        # tensor([[[0.4853, 0.5571, 0.5435, 0.1544],
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


        # 첫번째 hidden state을 0 벡터로 초기화
        #                        1         배치크기(20)           4
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)
        
        # 리턴값 outputs는 모든 시점 (timesteps)의 hidden state들이며 last_hs는 마지막 시점(timestep)의 hidden state를 나타냄
        # 즉 outputs은 [ h1, h2, h3, h4, last_hs ] 임

        outputs, (hidden_state) = self.gru(data, h0)

        # print('outputs.shape, hidden_state.shape =', outputs.shape, hidden_state.shape)

        #sequence_length : 6일째 주가를 예측하기 위한 5일간의 데이터

        # hidden_state.shape = torch.Size([1, 20, 4]) num_layers, batch_size, hidden_size

        # print('hidden_state =', hidden_state)

        # hidden_state = tensor([[[-0.6665, 0.7019, 0.2562, 0.6224],
        #                         [-0.6438, 0.6781, 0.2636, 0.6353],
        #                         [-0.6296, 0.6654, 0.2673, 0.6335],
        #                         [-0.6199, 0.6552, 0.2685, 0.6404],
        #                         [-0.6233, 0.6615, 0.2672, 0.6493],
        #                         [-0.6297, 0.6773, 0.2628, 0.6671],
        #                         [-0.6267, 0.6731, 0.2614, 0.6838],
        #                         [-0.6231, 0.6580, 0.2626, 0.6782],
        #                         [-0.6152, 0.6534, 0.2629, 0.6873],
        #                         [-0.6333, 0.6717, 0.2594, 0.6665],
        #                         [-0.6306, 0.6762, 0.2584, 0.6931],
        #                         [-0.6306, 0.6753, 0.2575, 0.7028],
        #                         [-0.6285, 0.6682, 0.2572, 0.7111],
        #                         [-0.6314, 0.6709, 0.2564, 0.7171],
        #                         [-0.6302, 0.6710, 0.2548, 0.7169],
        #                         [-0.6271, 0.6627, 0.2562, 0.7122],
        #                         [-0.6311, 0.6701, 0.2554, 0.6990],
        #                         [-0.6202, 0.6543, 0.2586, 0.7023],
        #                         [-0.6179, 0.6516, 0.2600, 0.7015],
        #                         [-0.6311, 0.6739, 0.2566, 0.7041]]],
        #                       grad_fn= < StackBackward0 >)

        # outputs.shape = torch.Size([20, 5, 4])      batch_size, sequence_length, input_size

        # [20, 5, 4] 에서 가운데 5에서 마지막 값을 가져옴
        last_hs = outputs[:, -1, :]
        prediction = self.fc(last_hs)

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

        # print('prediction =\n',prediction )

        # prediction =
        # tensor([[0.4975],
        #         [0.5034],
        #         [0.5109],
        #         [0.5267],
        #         [0.5255],
        #         [0.5268],
        #         [0.5330],
        #         [0.5298],
        #         [0.5222],
        #         [0.5188],
        #         [0.5118],
        #         [0.5165],
        #         [0.5319],
        #         [0.5250],
        #         [0.5143],
        #         [0.5154],
        #         [0.5026],
        #         [0.5079],
        #         [0.5029],
        #         [0.4885]], grad_fn= < AddmmBackward0 >)

        # print('6-6. class MyGRUModel(nn.Module) forward end')

        return prediction

        # print('6-7. class MyGRUModel(nn.Module) 선언 end ')

# print('6-8. class MyGRUModel(nn.Module) 정의 end ')

################################################################################
# 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
################################################################################

# print('7-1. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')
#                         4          4          1
# HIDDEN_SIZE : GRU가 만드는 출력값
model = MyGRUModel(FEATURE_NUMS, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print('7-2. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의 ')

################################################################################
# 9. train 함수 정의
################################################################################

# print('9-1. train 함수 정의 start')

def model_train(dataloader, model, loss_function, optimizer):

    # print('9-2. train 함수 정의 def start')

    model.train()  # 신경망을 학습모드로 전환

    train_loss_sum = 0

    total_train_batch = len(dataloader)

    for inputs, labels in dataloader:  # inputs에는 open, high, low, volume, labels에는 close

        x_train = inputs.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)               # 입력 데이터에 대한 예측값 계산(6일째 종가 계산)
        loss = loss_function(outputs, y_train) # 모델 예측값과 정답간의 오차(loss)인 손실함수 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_avg_loss = train_loss_sum / total_train_batch  # 학습 데이터 평균 오차 계산

    # print('9-3. train 함수 정의 def end')

    return train_avg_loss

# print('9-4. train 함수 정의 end')

################################################################################
# 10. 평가 함수 정의
################################################################################

# print('10-1. 평가 함수 정의 start')

def model_evaluate(dataloader, model, loss_function, optimizer):

    # print('10-2. 평가 함수 정의 def start')

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

    # print('10-3. 평가 함수 정의 def end')

    return val_avg_loss

# print('10-4. 평가 함수 정의 end')

################################################################################
# 8. 메인 처리
################################################################################

# print('8-1. 메인 처리 start ')

from datetime import datetime

train_loss_list = []

start_time = datetime.now()

EPOCHS = 200

for epoch in range(EPOCHS):

    avg_loss = model_train(train_loader, model, loss_function, optimizer)

    train_loss_list.append(avg_loss)

    if (epoch % 10 == 0):
        print('epoch: ', epoch, ', train loss = ', avg_loss)

end_time = datetime.now()

print('elapsed time => ', end_time-start_time)

# print('8-2. 메인 처리 end')

################################################################################
# train data 학습 결과
################################################################################

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
# ################################################################################
# # test data 예측과 실제 데이터 비교 (스케일링 되기 전의 원래값으로 변환 후)
# ################################################################################
#
# # print('11-1. test_pred_tensor = model(x_test_tensor) start')
#
# test_pred_tensor = model(x_test_tensor)                    # 328 개 test 데이터에 대한 예측 수행
#
# # print('11-2. test_pred_tensor = model(x_test_tensor) end')
#
#
# test_pred_numpy = test_pred_tensor.cpu().detach().numpy()  # 예측값을 numpy로 변환
#
# pred_inverse = scaler_y.inverse_transform(test_pred_numpy) # 스케일링 되기 전의 원래값으로 변환
#
# y_test_numpy = y_test_tensor.cpu().detach().numpy()        # 정답을 numpy로 변환
#
# y_test_inverse = scaler_y.inverse_transform(y_test_numpy)  # 스케일링 되기 전의 원래값으로 변환
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
# ################################################################################
# # test data 예측과 실제 데이터 비교 (스케일링 되기 전의 원래값으로 변환 전)
# ################################################################################
#
# import matplotlib.pyplot as plt
#
# plt.plot(y_test_numpy, label='actual')
# plt.plot(test_pred_numpy, label='prediction')
# plt.grid()
# plt.legend()
#
# plt.show()

##############################################################################
# 프로그램 처리 순서
# 1. 순수 함수(클래스가 없는)는 실제 호출되는 경우에만 실행함
# 2. 클래스 def __init__ 함수는 객체가 생성되는 단계에서 실행함(model = MyGRUModel() 객체가 생성되면 def __init__을 실행함)
# 3. 클래스 def forword  함수는 객체가 실행되는 단계에서 실행함(메인에서 model_train() 함수 호출 -> model_train() 함수에서 outputs = model(x_train)) 호출
##############################################################################
# 1-1.데이터 불러오기
# 1-2.데이터 불러오기
# 2-1.데이터 train, test 용 분리하기
# 2-2.데이터 train, test 용 분리하기
# 3-1.데이터 정규화 시작
# 3-2.데이터 정규화 끝

# 5-1. 순차적 Numpy Data 만들기 start  : def 문은 실행하지 않음
# 5-4. 순차적 Numpy Data 만들기 end    : def 문은 실행하지 않음

# 4-1. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 start
# 5-2. 순차적 Numpy Data 만들기 def start : 순차적 Numpy Data 만들기 함수 호출 시에 def(함수)가 실행됨 (train data)
# 5-3. 순차적 Numpy Data 만들기 def end   : 순차적 Numpy Data 만들기 함수 호출 시에 def(함수)가 실행됨 (train data)
# 5-2. 순차적 Numpy Data 만들기 def start : 순차적 Numpy Data 만들기 함수 호출 시에 def(함수)가 실행됨 (test data)
# 5-3. 순차적 Numpy Data 만들기 def end   : 순차적 Numpy Data 만들기 함수 호출 시에 def(함수)가 실행됨 (test data)
# 4-2. 순차적 Numpy Data 만들기 -> 텐서 데이터셋으로 변환 end

# 6-1. class MyGRUModel(nn.Module) 정의 start
# 6-2. class MyGRUModel(nn.Module) 선언 start  : 문장 순서로 class를 지나가며 class 문 안까지 들어가나 def __init__ def_forward 영역은 실행하지 않음
# 6-8. class MyModel(nn.Module) 정의 end   : 문장 순서로 class를 지나가며 class 문 안까지 들어가나 def __init__ def_forward 영역은 실행하지 않음

# 7-1. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의
# 6-3. class MyGRUModel(nn.Module) __init__ start : model = MyGRUModel() 객체가 생성되면 def __init__을 실행함
# 6-4. class MyGRUModel(nn.Module) __init__ end   : model = MyGRUModel() 객체가 생성되면 def __init__을 실행함
# 7-2. 7. 객체 생성, 손실함수 정의, 옵티마이저 정의

# 9-1. train 함수 정의 start : 문장 순서대로 train 함수안에 들어가지 않고 밖에서 지나감
# 9-4. train 함수 정의 end   : 문장 순서대로 train 함수안에 들어가지 않고 밖에서 지나감

# 10-1. 평가 함수 정의 start  : 문장 순서대로 평가 함수안에 들어가지 않고 밖에서 지나감
# 10-4. 평가 함수 정의 end    : 문장 순서대로 평가 함수안에 들어가지 않고 밖에서 지나감

# 8-1. 메인 처리 start

# 9-2. train 함수 정의 def start   : 메인에서 model_train 함수를 호출하면 train 함수가 실행됨
# 6-5. class MyModel(nn.Module) forward start  : model_train에서 model를 호출하면 class MyModel(nn.Module) forward 문이 실행함
# 6-6. class MyModel(nn.Module) forward end
# 9-3. train 함수 정의 def end

# 8-2. 메인 처리 end

# 11-1. test_pred_tensor = model(x_test_tensor) start
# 6-5. class MyModel(nn.Module) forward start
# 6-6. class MyModel(nn.Module) forward end
# 11-2. test_pred_tensor = model(x_test_tensor) end