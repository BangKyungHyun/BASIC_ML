# 9.2.5정규화(Normalization)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 데이터셋을 훈련과 테스트 용도로 분리하기 위한 라이브러리
from sklearn.model_selection import train_test_split
# 정규화와 관련된 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 경로 지정 및 훈련과 테스트 용도로 분리
df = pd.read_csv('data/diabetes.csv')
X = df[df.columns[:-1]]   # 여덟 개의 컬럼은 당뇨병을 예측하는 사용
y = df['Outcome']         # 당뇨병인지 아닌지 나타내는 레이블(정답)

X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# 훈련과 테스트용 테이터를 정류화
ms = MinMaxScaler()
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
y_train =y_train.reshape(-1, 1) # (?,1) 의 형태를 갖도록 변경. 즉 열의 수만 1로 고정
y_train = ms.fit_transform(y_train)

X_test = ss.fit_transform(X_test)
y_test = y_test.reshape(-1, 1)
y_test = ms.fit_transform(y_test)

# 커스텀 데이터셋 생성
class customdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

# 데이터 로더에 데이터 담기
train_data = customdataset(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
test_data = customdataset(torch.FloatTensor(X_test),torch.FloatTensor(y_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# 네트워크 생성
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(8, 64, bias=True) # 컬럼이 여덟개이므로 입력 크기는 8을 사용
        self.layer_2 = nn.Linear(64, 64, bias=True)
        self.layer_out = nn.Linear(64, 1, bias=True) # 출력으로 당뇨인지 아닌지를 나타내는 0과 1의 값만 가지므로 1를 사용
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

# 손실함수와 옵티마이저 지정
epochs = 1000+1
print_epoch = 100
LEARNING_RATE = 1e-2

model = binaryClassification()
model.to(device)
print(model)
BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 모델 성능 측정 함수 정의
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    # 실제 정답과 모델의 결과가 일치하는 갯수를 실수 형태로 변수에 저장
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

# 모델 학습
for epoch in range(epochs):
    iteration_loss = 0.     # 변수를 0으로 초기화
    iteration_accuracy = 0.

    model.train()  # 모델 학습
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device) # 독립 변수를 모델에 적용하여 훈련
        # 모델에 적용하여 훈련시킨 결과 정답 레이브를 손실함수에 적용
        loss = BCE(y_pred, y.reshape(-1, 1).float())

        iteration_loss += loss # 오차 값을 변수에 누적하여 저장
        iteration_accuracy += accuracy(y_pred, y) # 모델성능(정확도)을 변수에 누적하여 저장
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch % print_epoch == 0):
        print('Train: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch,
                                                                       iteration_loss / (
                                                                                   i + 1),
                                                                       iteration_accuracy / (
                                                                                   i + 1)))
    iteration_loss = 0.
    iteration_accuracy = 0.
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
    if (epoch % print_epoch == 0):
        print('Test: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch,
                                                                      iteration_loss / (
                                                                                  i + 1),
                                                                      iteration_accuracy / (
                                                                                  i + 1)))