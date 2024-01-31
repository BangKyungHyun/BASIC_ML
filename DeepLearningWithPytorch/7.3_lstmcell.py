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
# download_root : MNIST를 내려 받을 위치 지정
# transform=mnist_transform : 앞 단계 데이터 전처리 적용
# train : True로 설정할 경우 훈련용 데이터, False는 테스트용 데이터를 가져옴
# download : True로 설정할 경우 내려 받으려는 위치에 MNIST 파일이 없으면 내려받지만
#            파일이 있다면 내려 받지 않음
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
print('len(train_dataset) = ',len(train_dataset))
print('len(valid_dataset) = ',len(valid_dataset))
print('len(test_dataset) = ',len(test_dataset))

# len(train_dataset) =  60000
# len(valid_dataset) =  10000
# len(test_dataset) =  10000
################################################################################
# 변수 값 지정
################################################################################

batch_size = 100
n_iters = 6000
# num_epochs = 6000 / (60000 / 100) = 6000/600  = 10
num_epochs = n_iters / (len(train_dataset) / batch_size)

num_epochs = int(num_epochs)

print('num_epochs = ',num_epochs)

################################################################################
# LSTM 셀 네트워크 구축
################################################################################

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # 은닉층이 4개로 쪼개지는 상황이기 때문에 4를 곱함
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    # 모델의 파라미터 초기화
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std) # -std, std 사이의 임의의 실수인 난수 발생

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        # troch.squeeze는 텐서를 차원을 줄일 때 사용
        gates = gates.squeeze()

        # torch.chunk는 텐서를 쪼갤 때 사용하는 함수
        # 첫번째 파라미터 : 턴세를 몇 개로 쪼갤지 설정
        # 두번째 파라미터 : 어떤 차원을 기준을 쪼갤지를 설정. dim=이므로 열 단위로 텐서를 분할
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)         # 입력 게이트에서 시그모이드 활성화 함수 적용
        forgetgate = F.sigmoid(forgetgate) # 망각 게이트에서 시그모이드 활성화 함수 적용
        cellgate = F.tanh(cellgate)        # 셀 게이트에서 탄젠트 활성화 함수 적용
        outgate = F.sigmoid(outgate)       # 출력 게이트에서 시그모이드 활성화 함수 적용

        # 하나의 LSTM 셀을 통과하면 셀(Ct) 상태와 은닉 상태(Ht)가 출력으로 주어짐
        # 이 때 셀 상태는 입력,망각, 셀 게이트에 의해 계산되며, 은닉상태는 출력 게이트에 의해 계산
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return (hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available():
            c0 = Variable(
                torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), hidden_dim))

        outs = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


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

seq_dim = 28
loss_list = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            loss.cuda()

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in valid_loader:
                if torch.cuda.is_available():
                    images = Variable(
                        images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter,
                                                                 loss.item(),
                                                                 accuracy))


def evaluate(model, val_iter):
    corrects, total, total_loss = 0, 0, 0
    model.eval()
    for images, labels in val_iter:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim)).to(device)

        logit = model(images).to(device)
        loss = F.cross_entropy(logit, labels, reduction="sum")
        _, predicted = torch.max(logit.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy

test_loss, test_acc = evaluate(model,test_loader)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))