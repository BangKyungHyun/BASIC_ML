# ################################################################################
# # 2.2.1 텐서 다루기
# ################################################################################
#
# #-------------------------------------------------------------------------------
# # 텐서 생성 및 변환
# # 2차원 형태의 텐서 생성
# #-------------------------------------------------------------------------------
#
# import torch
#
# print(torch.tensor([[1,2],[3,4]]))
# # tensor([[1, 2],
# #         [3, 4]])
#
# #print(torch.tensor([[1,2],[3,4]], device="cuda:0")) #GPU가 없다면 오류가 발생하므로 주석 처리하였습니다.
# # dtype을 이용하여 텐서 생성
#
# print(torch.tensor([[1,2],[3,4]], dtype=torch.float64))
# # tensor([[1., 2.],
# #         [3., 4.]], dtype=torch.float64)
#
# # 텐서를 narray로 변환
# temp = torch.tensor([[1,2],[3,4]])
# print(temp.numpy())
# # [[1 2]
# #  [3 4]]
#
# #temp = torch.tensor([[1,2],[3,4]], device="cuda:0") #GPU가 없다면 오류가 발생하므로 주석 처리하였습니다.
# temp = torch.tensor([[1,2],[3,4]], device="cpu:0")
# #GPU상의 텐서를 CPU의 텐서로 변환 한 후 ndarray로 변환
# # ndarray : N차원 배열 객체, 하나이 데이터 타입만 가능
# print(temp.to("cpu").numpy())
# # [[1 2]
# #  [3 4]]
#
# # 텐서의 인덱스 조작
# # 파이토치로 1차원 벡터 생성
# temp = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
# # 인덱스로 접근
# print(temp[0], temp[1], temp[-1])
# # tensor(1.) tensor(2.) tensor(7.)
#
# # 슬라이스로 접근
# print(temp[2:5], temp[4:-1])
# # tensor([3., 4., 5.]) tensor([5., 6.])
#
# # 길이가 3인 벡터 생성
# v = torch.tensor([1, 2, 3])
# w = torch.tensor([3, 4, 6])
#
# # 길이가 같은 벡터간 뺄셈 연산
# print(w - v)
# # tensor([2, 2, 3])
#
# # 텐서 연산 및 차원 조작
# # 2*2 행렬 생성
# temp = torch.tensor([[1, 2], [3, 4]])
# print(temp.shape)
# # torch.Size([2, 2])
#
# # 2*2 행렬을 4*1 행렬로 변형
# print(temp.view(4,1))
#
# # tensor([[1],
# #         [2],
# #         [3],
# #         [4]])
#
# print(temp.view(-1))
# # tensor([1, 2, 3, 4])
#
# print(temp.view(1, -1))
# # tensor([[1, 2, 3, 4]])
# # -1은 (1,?)와 같은 의미로 다른 차원으로 부터 해당 값을 유추하겠다는 것입니다.
# # temp의 원소 갯수 (2*2=4)를 유지한채 (1,?)의 형태를 만족해야 하므로 (1,4)가 됩니다.
#
# print(temp.view(-1, 1))
# # 앞에서와 마찬가지로 (?,1)의 의미로 temp 원소 갯수(2*2=4)를 유지한 채 (?,1)의
# # 형태를 만족해야 하므로 (4,1)가 됩니다.

#2.4 파이토치 코드 맛보기

import torch
import torch.nn as nn
# 벡터 및 행렬 연산 라이브러리
import numpy as np
# 데이터 처리 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
# 시각화 차트를 제공하는 라이브러리
import seaborn as sns
# %matplotlib inline

# file_uploaded=files.upload()   # 데이터 불러오기
dataset = pd.read_csv('data/car_evaluation.csv')

print(dataset.head())
#    price  maint doors persons lug_capacity safety output
# 0  vhigh  vhigh     2       2        small    low  unacc
# 1  vhigh  vhigh     2       2        small    med  unacc
# 2  vhigh  vhigh     2       2        small   high  unacc
# 3  vhigh  vhigh     2       2          med    low  unacc
# 4  vhigh  vhigh     2       2          med    med  unacc

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot \
    (kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05, 0.05, 0.05,0.05))

################################################################################
# 데이터를 범주형 타입으로 변환
################################################################################

# 예제 데이터셋 컬럼들의 목록
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']

# astype() 메서드를 이용하여 데이터를 범주형으로 전환
# 범주형 데이터를 텐서로 변환하기 위해서는 다음과 같은 절차가 필요

# 범주형 데이터 -> dataset(category) -> 넘파이 배열 -> 텐서
# 범주형 데이터(단어)를 넘파이 배열(숫자)로 변환하기 위해 cat.codes를 사용

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')

price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

# print("price = ", price)
# # price =  [3 3 3 ... 1 1 1]

# np.stack은 두 개 이상의 넘파이 객체를 합칠 때 이용
categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
# print('categorical_data[:10] = \n ',categorical_data[:10])  # 합친 넘파이 배열 중 열개의 행을 출력하여 보여 줌
#
# # categorical_data[:10] =
# #   [[3 3 0 0 2 1]
# #  [3 3 0 0 2 2]
# #  [3 3 0 0 2 0]
# #  [3 3 0 0 1 1]
# #  [3 3 0 0 1 2]
# #  [3 3 0 0 1 0]
# #  [3 3 0 0 0 1]
# #  [3 3 0 0 0 2]
# #  [3 3 0 0 0 0]
# #  [3 3 0 1 2 1]]

################################################################################
# 배열을 텐서로 변환
################################################################################

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
print('categorical_data[:10] = \n' , categorical_data[:10])
# categorical_data[:10] =
#  tensor([[3, 3, 0, 0, 2, 1],
#         [3, 3, 0, 0, 2, 2],
#         [3, 3, 0, 0, 2, 0],
#         [3, 3, 0, 0, 1, 1],
#         [3, 3, 0, 0, 1, 2],
#         [3, 3, 0, 0, 1, 0],
#         [3, 3, 0, 0, 0, 1],
#         [3, 3, 0, 0, 0, 2],
#         [3, 3, 0, 0, 0, 0],
#         [3, 3, 0, 1, 2, 1]])

################################################################################
# 레이블로 사용한 컬럼을 텐서로 변환
################################################################################

#pd.get_dummies 는 가변수를 만들 수 있는 함수
# 가변수로 만들어 준다는 의미는 문자를 숫자(0,1)로 바꾸어 준다는 의미
outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten() # 1차원 텐서로 변환

print(categorical_data.shape)
print(outputs.shape)

# torch.Size([1728, 6])
# torch.Size([6912])

################################################################################
# 범주형 컬럼을 N차원으로 변환
################################################################################

categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)
# 모든 범주형 컬럼의 고유 값 수, 차원의 크기 형태의 배열을 출력한 결과
# [(4, 2), (4, 2), (4, 2), (3, 2), (3, 2), (3, 2)]

################################################################################
# 데이터셋 분리
################################################################################

total_records = 1728
test_records = int(total_records * .2) # 전체 데이터 중 20%을 테스트용으로 사용
# 전체 행 : 80% 열
categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

print(categorical_train_data.shape)
print(categorical_test_data.shape)

print(len(categorical_train_data))
print(len(categorical_test_data))

print(len(train_outputs))
print(len(test_outputs))

# 1383
# 1383
# 345
# 345

################################################################################
# 모델의 네트워크 생성
################################################################################

# 클래스 형태로 구현되는 모델은 nn.Module을 상속 받음
class Model(nn.Module):
#__init__()은 모델에서 사용될 파라미터와 신경만을 초기화하기 위한 용도로 사용로
# 객체가 생성될 때 자동으로 생성
# 1) self : 첫번째 파라미터는 self를 지정해야 하며 자기자신을 의미
# 2) embedding_size : 범주형 컬럼의 임베딩 크기
# 3) output_size : 출력층의 크기
# 4) layers : 모든 계층에 대한 목록
# 5) p=0.4 : 드롭아웃

    def __init__(self, embedding_size, output_size, layers, p=0.4):
        # super().__init__()은 부모 클래스(Model 클래스)에 접근할 때 사용
        # super는 self를 사용하는 않은 것에 주의해야 함
        super().__init__()

        self.all_embeddings = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        # 입력층의 크기를 찾기 위해 범주형 컬럼 갯수를 input_size 변수에 저장
        input_size = num_categorical_cols

        # 모델의 네트워크 계층을 구축하기 위해 for문을 이용하여 각 계층을 all_layers 목록에 추가
        # Linear : 선형계층은 입력 데이터에 선현 변형을 진행한 결과. y = Wx+b
        # ReLU : 활성화 함수로 사용
        # BatchNorm1d : 배치 정규화 용도로 사용
        # Dropout : 과적합 방지에 사용
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        # 신경망의 모든 계층이 순차적으로 실행되도록 모든 계층에 대한 목록(all_layers)을
        # nn.sequential 클래스로 전달
        self.layers = nn.Sequential(*all_layers)

    # forward() 함수는 학습 데이터를 입력 받아서 연산을 진행.forward()함수는 모델 객체를
    # 데이터와 함께 호출하면 자동으로 실행
    def forward(self, x_categorical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x

################################################################################
# 모델 클래스의 객체 생성
################################################################################

# 모델 훈련을 위해 앞에서 정의했던 Model 클래스의 객체를 생성
# 객체를 생성하면서(범주형 컬럼의 임베딩 크기, 출력 크기, 은닉층의 뉴런, 드롭아웃)을 전달
# 은닉층을 [200,100,50]로 정의

model = Model(categorical_embedding_sizes, 4, [200,100,50], p=0.4)
print(model)

# Model(
#   (all_embeddings): ModuleList(
#     (0-2): 3 x Embedding(4, 2)
#     (3-5): 3 x Embedding(3, 2)
#   )
#   (embedding_dropout): Dropout(p=0.4, inplace=False)
#   (layers): Sequential(
#     (0): Linear(in_features=12, out_features=200, bias=True)
#     (1): ReLU(inplace=True)
#     (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (3): Dropout(p=0.4, inplace=False)
#     (4): Linear(in_features=200, out_features=100, bias=True)
#     (5): ReLU(inplace=True)
#     (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (7): Dropout(p=0.4, inplace=False)
#     (8): Linear(in_features=100, out_features=50, bias=True)
#     (9): ReLU(inplace=True)
#     (10): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (11): Dropout(p=0.4, inplace=False)
#     (12): Linear(in_features=50, out_features=4, bias=True)
#   )
# )

################################################################################
# 모델의 파라미터 정의
################################################################################

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

################################################################################
# CPU/GPU 사용 지정
################################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

################################################################################
# 모델 학습
################################################################################

epochs = 5000
aggregated_losses = []
train_outputs = train_outputs.to(device=device, dtype=torch.int64)

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i%100 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

################################################################################
# 테스트 테이터셋으로 모델 예측
################################################################################

test_outputs = test_outputs.to(device=device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')

# epoch: 4901 loss: 0.43003395
# epoch: 5000 loss: 0.4354535043

################################################################################
# 모델의 예측 확인
################################################################################

print(y_val[:5])
#
# Loss: 0.47946301
################################################################################
# 가장 큰 값을 가진 인덱스 확인
################################################################################
y_val = np.argmax(y_val, axis=1)
print(y_val[:5])

# tensor([0, 0, 0, 0, 0])

################################################################################
# 테스트 데이터셋을 이용한 정확도 확인
################################################################################

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_outputs,y_val))
print(classification_report(test_outputs,y_val))
print(accuracy_score(test_outputs, y_val))

# [[241  18]
#  [ 57  29]]
#               precision    recall  f1-score   support
#
#            0       0.81      0.93      0.87       259
#            1       0.62      0.34      0.44        86
#
#     accuracy                           0.78       345
#    macro avg       0.71      0.63      0.65       345
# weighted avg       0.76      0.78      0.76       345

# 0.782608695652174