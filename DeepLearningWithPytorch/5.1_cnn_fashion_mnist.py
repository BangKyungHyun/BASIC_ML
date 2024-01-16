# 5.2 합성곱 신경망 맛보기
################################################################################
# 라이브러리 호출
################################################################################
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms # 데이터 전처리를 위해 사용하는 라이브러리
from torch.utils.data import Dataset, DataLoader

################################################################################
# CPU 혹은 GPU 장치 확인
################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################
# fashion_mnist 데이터셋 내려받기
# 28*28 픽셀의 7만개 데이터
# train_dataset는 6만개를 100개 단위로 읽으면 600번 반복함
# test_dataset는 1만개를 100개 단위로 읽으면 100번 반복함
################################################################################
train_dataset = torchvision.datasets.FashionMNIST("data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))
                                               # transform : 이미지를 텐서(0~1)로 변경

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100)

################################################################################
# 분류에 사용될 클래스 정의
################################################################################
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

fig = plt.figure(figsize=(8,8));  # 출력할 이미지의 가로세로 길이로 단위는 inch
columns = 4;
rows = 5;

for i in range(1, columns*rows +1):
    # np.random.randint 는 이산형 분포를 작는 데이터에서 무작위 표본을 추출할 때 사용
    # np.random.randint(len(train_dataset)) 의미는 0~(train_dataset의 길이) 값을
    # 값을 분포에서 랜던한 숫자 한개를 생성하라는 의미
    img_xy = np.random.randint(len(train_dataset));
    # print('img_xy = np.random.randint(len(train_dataset)) = ', img_xy)
    # img_xy = np.random.randint(len(train_dataset)) = 2027
    # img_xy = np.random.randint(len(train_dataset)) = 3836
    # img_xy = np.random.randint(len(train_dataset)) = 57951
    # img_xy = np.random.randint(len(train_dataset)) = 55755
    # img_xy = np.random.randint(len(train_dataset)) = 26504
    # img_xy = np.random.randint(len(train_dataset)) = 7164
    # img_xy = np.random.randint(len(train_dataset)) = 20694
    # img_xy = np.random.randint(len(train_dataset)) = 21199
    # img_xy = np.random.randint(len(train_dataset)) = 55140
    # img_xy = np.random.randint(len(train_dataset)) = 43997
    # img_xy = np.random.randint(len(train_dataset)) = 37649
    # img_xy = np.random.randint(len(train_dataset)) = 59170
    # img_xy = np.random.randint(len(train_dataset)) = 44321
    # img_xy = np.random.randint(len(train_dataset)) = 11465
    # img_xy = np.random.randint(len(train_dataset)) = 26321
    # img_xy = np.random.randint(len(train_dataset)) = 59841
    # img_xy = np.random.randint(len(train_dataset)) = 21096
    # img_xy = np.random.randint(len(train_dataset)) = 54166
    # img_xy = np.random.randint(len(train_dataset)) = 42633
    # img_xy = np.random.randint(len(train_dataset)) = 34913

    # train_dataset을 이용한 3차원 배열 생성
    # train_dataset[img_xy][0][0, :, :] 의미는 train_dataset에서
    # train_dataset[img_xy][0][0, :, :] 에 해당하는 요소 값을 가져오겠다는 의미

    img = train_dataset[img_xy][0][0,:, : ]
    # print('img = train_dataset[img_xy][0][0,:,:] = ', img)
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
# plt.show()

################################################################################
# 심층 신경망 모델 생성
################################################################################

# 클래스(class) 형태의 모델은 항상 torch.nn.Module을 상속 받습니다. 
class FashionDNN(nn.Module):
    #__init__(self) 부분은 객체가 갖는 속성 값을 초기화하는 역할을 함
    # 객체가 생성될 때 자동으로 호출됨
    def __init__(self):
        # super(FashionDNN,self).__init__() 는 FashionDNN 이라는 부모 클래스를 상속
        # 받겠다는 의미임
        super(FashionDNN,self).__init__()
        self.fc1 = nn.Linear(in_features=784,out_features=256)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=10)

    def forward(self,input_data):
        # view는 넘파이의 reshape과 같은 역할로 텐서의 크기(shape)를 변경하는 역할
        # input_data.view(-1, 784)는 input_data를 (?, 784)의 크기로 변경하라는 의미
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

################################################################################
# 심층 신경망에서 필요한 하이퍼 파라미터 정의
################################################################################
learning_rate = 0.001;
model = FashionDNN();
model.to(device)

criterion = nn.CrossEntropyLoss(); # 분류 문제에서 사용할 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);
print('# 심층 신경망 모델 #')
print(model)

################################################################################
# 심층 신경망을 이용한 모델 학습
################################################################################
num_epochs = 5
count = 0
# 배열이나 행렬과 같은 리스트를 사용하는 방법
# 비어 있는 배열이나 행렬을 만들고 append 메서드를 이용하여 데이터를 하나씩 추가함
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # autograd는 Variable을 사용해서 역전파을 위한 미분 값을 자동으로 계산
        # 따라서 자동 미분을 계산하기 위해 torch.autograd 패키지 안에 있는 Variable을 이용
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        # 학습 데이터를 모델에 적용
        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        # if epoch == 0:
        #     print(' count = ', count)

        if count % 50 == 0:
            # print(' 1) count 2) epoch, = ', count,  epoch)

            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if count % 500 == 0:  # true
            # print(' 1) count 2)  epoch, = ', count, epoch)

            print("DNN Iteration: {}, Loss: {}, Accuracy: {}%".format(count,
                                                                  loss.data,
                                                                  accuracy))


################################################################################
# 합성곱 네트워크 생성
################################################################################

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # nn.sequential을 사용하면 __init__()에서 사용할 네트워크 모델들을 정의해 줄 뿐만 아니라
        # forward() 함수에서 구현될 순전파를 계층 형태로 좀 더 가독성이 뛰어난 코드로 작성
        self.layer1 = nn.Sequential(
            # 합성곱층은 합성곱 연산을 통해서 이미지 특징을 추출함
            # 합성곱이란 커널(필터)이라의 N*M 크기의 행렬이 높이와 너비 크기의 이미지를 처음부터
            # 끝까지 훍으면서 각 원소 값끼리 곱한 후 모두 더한 값을 출력
            # in_channels : 입력 채널의 수를 의미 - 흑백 이미지는 1, RGB값 이미지는 3개를 가짐
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            #-------------------------------------------------------------------
            # Conv2d 계층에서의 출력 크기 구하는 방식
            # 출력크기 : (W-F+2P)S+1
            # W : 입력 데이터의 크기 - fashion_mnist 입력 데이터 크기 784 (28*28)
            # F : 커널 크기
            # P : 패딩 크기
            # S : 스트라이드 - 스트라이드가 명시되지 않지 않다면 기본값은 (1,1)
            # (784-3+(2*1) /1 + 1 = 784 
            # 계산 결과를 적용하면 출력의 형태는 [32,784,784] 
            #-------------------------------------------------------------------
            # BatchNorm2d는 학습과정에서 각 배치 단위별로 데이터가 다양한 분포를 가지더라도
            # 평균과 분산을 이용하여 정규화하는 것을 의미
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # MaxPool2d 는 이미지를 축소시키는 용도로서 합성곱층의 출력 데이터를 입력으로 받아서
            # 출력데이터 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용
            nn.MaxPool2d(kernel_size=2, stride=2)
            #-------------------------------------------------------------------
            # MaxPool2d 계층에서의 출력 크기 구하는 방식
            # 출력 크기 : IF / F
            # IF : 입력 데이터의 크기(input_filter_size, 또한 바로 앞의 Conv2d의 출력크기)
            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # 784-3+(2*1)/1+1 = 784 ==> [64, 784,784]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 클래스를 분류하기 위해서는 이미지 형태의 데이터를 배열 형태로 변환하여 작업해야 함
        # 이 때 Conv2d에서 사용하는 하이퍼파라미터 값들에 따라 출력크기가 달라짐
        # 즉 패딩과 스트라이드의 값에 따라 출력 크기가 달라짐
        # 이렇게 줄어든 출력 크기는 최종적으로 분류를 담당하는 완전연결층으로 전달됨
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # 합성곱층에서 완전연결층으로 변경되기 때문에 데이터의 형태를 1차원으로 바꾸어야 함
        # 이 때 out.size(0)은 결국 100을 의미함. 따라서 (100,?) 크기이 텐서로 변경하겠다는
        # 의미임 
        # out.view(out.size(0),-1) 에서 '-1'은 행의 수는 정확히 알고 있지만 열(컬럼)
        # 숫자를 정확하게 알지 못할 때 사용
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

################################################################################
# 합성곱 네트워크를 위한 파라미터 정의
################################################################################
learning_rate = 0.001;
model = FashionCNN();
model.to(device)

criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate);
print('# 합성곱 네트워크 ')
print(model)

num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("CNN Iteration: {}, Loss: {}, Accuracy: {}%".format(count,
                                                                  loss.data,
                                                                  accuracy))


