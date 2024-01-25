################################################################################
# 라이브러리 호출
################################################################################
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.autograd import Variable
from torch.nn import Parameter # 파라미터 목록을 작는 있는 라이브러리(패키지)
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math # 수학과 관련되어 다양한 함수들과 상수들이 정의되어 잇는 라이브러리

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GPU 사용에 필요
cuda = True if torch.cuda.is_available() else False
# GPU 사용에 필요
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

################################################################################
# 데이터 전처리
################################################################################

import torchvision.transforms as transforms
# 편균과 표준편차에 맞게 데이터를 정규화하기 휘한 코드
# 평균을 0.5, 표준편차를 1.0으로 데이터 정규화 (데이터 분포를 조정)
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

################################################################################
# 데이터셋 내려 받기
################################################################################

from torchvision.datasets import MNIST

download_root = '../chap07/MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

################################################################################
# 데이터셋을 메모리로 가져오기
################################################################################
batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)
valid_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

################################################################################
# 변수 값 지정
################################################################################

batch_size = 100
n_iters = 6000


print(' len(train_dataset) = ',len(train_dataset))
# len(train_dataset) = 60000

num_epochs = n_iters / (len(train_dataset) / batch_size)
           # =  6,000/ (60,000/100) = 6,000 / 600 = 10
num_epochs = int(num_epochs)
print('num_epochs =' , num_epochs)

################################################################################
# LSTM 셀 네트워크 구축
################################################################################

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    # 모델의 파라미터 초기화
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            # uniform()은 난수을 위해 사용 (from~to사이의 임의의 실수)
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        # troch.squeeze()는 텐서의 차원을 줄일 때 사용
        gates = gates.squeeze()
        # chunks는 텐서를 쪼갤 때 사용하는 함수
        # 첫번째 파라미터(4) : 텐서를 몇 개로 쪼갤지 설정
        # 두번째 파라미터(1) : 어떤 차원을 기준을 쪼갤지를 결정
        #                    dim=1이므로 열(컬럼) 단위로 텐서를 분할하겠다는 의미
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # torch.mul()은 텐서에 곱셈을 할 때 사용
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)

################################################################################
# LSTM 셀의 전반적인 네트워크
################################################################################

class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        # 은닉층의 뉴런/유닛 갯수
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM 쉘은 앞에서 정의한 함수를 불러오는 부분
        # input_dim : 입력에 대한 특성(feature) 수 (컬럼 갯수)
        # hidden_dim : 은닉층의 뉴런 갯수
        # layer_dim : 은닉층의 계층 갯수
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # GPU 사용 여부 확인
        if torch.cuda.is_available():
            h0 = Variable(
                # 은닉층의 계층 갯수, 배치크기, 은닉층의 뉴런 갯수 형태를 갖는 은닉상태을 0으로 초기화
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available():
            c0 = Variable(
                # 은닉층의 계층 갯수, 배치크기, 은닉층의 뉴런 갯수 형태를 갖는 셀상태을 0으로 초기화
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), hidden_dim))

        outs = []
        # (은닉층의 계층 갯수, 배치 크기, 은닉층의 뉴런 갯수) 크기를 갖는 셀 상태에 대하 텐서
        cn = c0[0, :, :]
        
        # (은닉층의 계층 갯수, 배치 크기, 은닉층의 뉴런 갯수) 크기를 갖는 은닉 상태에 대하 텐서
        hn = h0[0, :, :]

        # LSTM 셀 계층을 반복하여 쌓아 올림
        for seq in range(x.size(1)):
            # 은닉상태(hh)와 셀 상태를 LSTMCell에 적용한 결과를 또 다시 hh,cn에 저장
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out

################################################################################
# 옵티마이저와 손실 함수 지정
################################################################################

input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

################################################################################
# 모델학습과 성능 확인
################################################################################

seq_dim = 28
loss_list = []
iter = 0

for epoch in range(num_epochs):
    # 훈련용 데이터셋을 이용한 모델 학습
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():

            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        # 손실함수을 이용하여 오차 계산
        loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            loss.cuda()

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        iter += 1

        # 정확도 계산
        if iter % 500 == 0:
            correct = 0
            total = 0
            # 검증 데이터셋을 이요한 모델 성능 검증
            for images, labels in valid_loader:
                if torch.cuda.is_available():
                    images = Variable(
                        images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                outputs = model(images)
                # 모델을 통과한 최대값으로부터 예측 결과 가져오기
                _, predicted = torch.max(outputs.data, 1)

                # 총 레이블 수
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter,
                                                                 loss.item(),
                                                                 accuracy))

################################################################################
# 모델학습과 성능 확인
################################################################################
def evaluate(model, val_iter):
    corrects, total, total_loss = 0, 0, 0
    model.eval()
    for images, labels in val_iter:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim)).to(device)

        logit = model(images).to(device)

        # reduction='sum'을 지정했기 때문에 모든 오차를 더함
        loss = F.cross_entropy(logit, labels, reduction="sum")

        # logit.data 텐서에서 최대값의 인덱스를 반환
        _, predicted = torch.max(logit.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy

################################################################################
# 모델예측 성능 확인
################################################################################

test_loss, test_acc = evaluate(model,test_loader)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))