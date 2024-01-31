################################################################################
# 라이브러리 호출
################################################################################
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import datetime

################################################################################
# 데이터셋 내려받기 및 전 처리
################################################################################

train_dataset = datasets.MNIST(root='MNIST_data/', train=True,  # 학습 데이터
                               transform=transforms.ToTensor(),
                               # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                               download=True)

test_dataset = datasets.MNIST(root='MNIST_data/', train=False,  # 테스트 데이터
                              transform=transforms.ToTensor(),
                              # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                              download=True)

print(len(train_dataset))

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

print(len(train_dataset), len(validation_dataset), len(test_dataset))

################################################################################
# 데이터 불러오기 정의
################################################################################

# 배치 데이터를 만들기 위해 DataLoader 설정
BATCH_SIZE = 32

train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

################################################################################
# 모델 정의
################################################################################

class MyDeepLearningModel(nn.Module):
    # 아키텍쳐를 구성하는 다양한 계층 정의
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 10)

    # 입력을 주어진 학습 데이터에 대해서 피드포워드 수행
    def forward(self, data):
        data = self.flatten(data)  # 입력층
        data = self.fc1(data)      # 은닉층
        data = self.relu(data)     # ReLU(비선형 함수)
        data = self.dropout(data)  # Dropout
        logits = self.fc2(data)    # 출력층
        return logits

################################################################################
# 손실함수와 옵티마이저 지정
################################################################################

model = MyDeepLearningModel()

# CrossEntropyLoss 손실함수에는 softmax 함수가 포함되어 있음
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

################################################################################
# 모델 학습 함수(배치 size 단위로 하이퍼 파라미처 갱신함 )
################################################################################

def model_train(dataloader, model, loss_function, optimizer):

    # 신경망을 학습모드(모델 파라미터를 업데이터 하는 모드) 로 전환
    model.train()

    train_loss_sum = 0  # 학습 단계 손실값 합계
    train_correct = 0   # 학습 단계 정확도
    train_total = 0     # 학습 단계에서 사용한 데이터 건수
    for_looping_count = 0

    # 51,000개 train data를 32 batch size로 1593.75 번 나누어서 데이터를 로드함
    total_train_batch = len(dataloader)
    # print('len(dataloader) = ', len(dataloader))
    # len(dataloader) =  1594

    # images에는 이미지, labels에는 0-9 숫자
    for images, labels in dataloader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded

        for_looping_count += 1  # for 문 count

        x_train = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
        y_train = labels
        # print('lables = ', labels)
        # lables = tensor([0, 9, 6, 9, 5, 0, 9, 2, 6, 4, 3, 1, 5, 7, 5, 6, 2, 9, 7, 5, 0, 7, 2, 2, 7, 9, 9, 6, 8, 8, 6, 0])

        # 입력 데이터에 대한 예측값 계산 (softmax 값으로 표현됨)
        outputs = model(x_train)
        # print('outputs = ',outputs)
        # outputs = tensor(
        #     [[-2.5771e+00, 4.8128e+00, 6.7051e-01, 1.2016e+00, -1.6843e+00, 5.5225e-02, 3.3297e-01, -1.7632e+00, 1.0516e+00, -1.4326e+00],
        #      [5.4438e+00, -5.2946e+00, -2.6904e-01, 2.7745e-01, -8.1768e-01,3.2702e+00, 3.0684e-01, -1.6883e+00, 9.0942e-01, -6.7880e-01],
        #      [-8.1428e-01, -7.7218e-01, 8.6069e+00, 3.9265e-01, -2.3453e+00,-1.8639e+00, 1.2387e+00, -4.0886e+00, 3.0393e+00, -1.0612e+00],
        #      [-3.1052e+00, 1.4543e+00, -4.0392e-01, 2.0039e-01, 1.0788e+00, 2.8152e-01, 2.9633e-01, -1.0464e+00, 8.9088e-01, 9.3998e-01],
        #      [1.7741e-01, -2.1115e+00, -1.1478e+00, 2.2030e+00, -1.1842e+00, 2.4964e+00, -7.7119e-01, 1.5188e-01, 1.2320e+00, 1.3049e-01],
        #      [2.7275e-01, -2.4492e+00, 2.7713e+00, -2.8922e+00, 1.9523e+00,  -1.7011e+00, 5.0147e+00, -1.4409e+00, -1.3727e+00, -2.6566e-02],
        #      [-2.5409e+00, -3.2833e+00, -3.4211e+00, -1.2188e+00, 4.2883e+00, 2.2483e+00, -6.7452e-01, 1.4705e-01, 1.2124e+00, 3.2404e+00],
        #      [-8.1440e-01, 7.3601e-01, 5.0355e-01, 5.5227e+00, -3.5701e+00, 3.1797e+00, -4.4180e+00, -6.2260e-01, 2.2989e+00, -1.7337e+00],
        #      [-2.5296e+00, 4.3709e+00, 7.0154e-01, 2.2132e-01, -1.0181e+00, -3.9734e-01, -7.7576e-01, 2.1980e-01, 9.9540e-01, -6.8203e-01],
        #      [4.4546e-01, -1.0715e+00, 1.1767e-01, 3.5482e-01, -2.1388e-01, 1.0051e+00, 3.6316e-01, -3.9907e-01, -5.4766e-02, -5.6440e-01],
        #      [-2.3783e+00, 2.1615e+00, 7.3878e+00, 5.3390e-01, -2.1246e+00, -2.2160e+00, 2.0875e+00, -2.7947e+00, 9.3702e-01, -3.4022e+00],
        #      [-2.3698e+00, 2.0020e+00, 7.8093e-01, 1.8847e+00, -2.3908e+00, -8.8739e-02, -2.5450e+00, 2.0262e+00, 1.5388e+00, 8.8360e-01],
        #      [-8.6841e-01, -3.0206e+00, 1.1870e+00, -2.6459e+00, 2.8173e+00,-1.1052e+00, 4.0449e+00, -1.1348e+00, -9.2888e-01, 7.0191e-01],
        #      [-3.6121e+00, 5.7497e+00, 2.1598e+00, 1.0380e+00, -2.4956e+00, -3.1678e-01, -3.0476e-03, -1.2496e+00, 1.8860e+00, -1.7609e+00],
        #      [-2.0358e+00, -2.5895e-01, 1.2351e+00, 1.3656e+00, -1.6507e+00,-1.3895e+00, -3.7468e+00, 3.6759e+00, 2.3406e+00, 1.7998e+00],
        #      [-4.7181e-02, -1.6843e+00, -6.2789e-01, 5.0757e-01, -1.5919e+00,2.7390e+00, -1.4480e+00, -7.0031e-01, 3.3725e+00, 3.6743e-01],
        #      [5.9670e-02, -2.6538e+00, 1.0313e+00, -3.8519e+00, 3.3189e+00, 2.2232e-01, 4.4157e+00, -2.3423e+00, 2.9602e-02, 7.6383e-01],
        #      [-7.5722e-01, -1.9132e+00, 1.2639e+00, -1.8339e+00, 3.5382e-01, 8.2506e-01, 5.0576e+00, -2.9678e+00, 7.9507e-02, -8.0940e-01],
        #      [-2.0033e+00, -4.2125e+00, 1.7286e+00, -4.1493e-01, 3.6627e-01, 1.8698e+00, -1.0803e+00, -1.5248e+00, 5.9666e+00, 1.1516e+00],
        #      [-9.1101e-01, 1.6142e+00, 6.9093e-01, 1.8140e+00, -1.6334e+00,  7.2876e-01, -7.7783e-01, -1.0888e+00, 1.1789e+00, -1.1670e+00],
        #      [1.2869e+00, -2.8401e+00, 3.9223e+00, 2.2865e+00, -4.7631e-01, 3.8746e-01, -4.9622e-01, -1.8311e+00, 2.6450e-01, -1.8711e+00],
        #      [-1.3208e+00, -3.0128e+00, 6.8140e-01, -2.1610e+00, 4.1040e+00, -1.6571e+00, 1.6391e-01, 1.3052e+00, -6.6265e-01, 2.4344e+00],
        #      [-1.2327e+00, 1.6387e+00, 3.1888e+00, 1.9173e+00, -1.7898e+00,-6.6698e-01, -8.4570e-01, -1.2951e+00, 2.1895e+00, -2.1763e+00],
        #      [9.4039e-01, -2.3702e+00, 6.1276e+00, -6.7124e-01, -1.2842e-01,-1.6586e+00, 1.8133e+00, -2.2338e+00, -3.1091e-01, -1.8765e+00],
        #      [-1.4212e+00, 1.7927e+00, -1.3205e-01, 3.3968e-01, -9.5928e-01,-1.9656e-01, -1.6011e+00, 2.2099e+00, 1.1846e-01, 5.7681e-01],
        #      [-2.1440e+00, -1.1820e+00, -2.6465e+00, 5.3738e-01, 4.1915e-01, 1.6572e+00, -1.9676e+00, 1.3225e+00, 9.5674e-01, 3.0416e+00],
        #      [-2.4889e+00, -2.2065e+00, 3.2395e+00, 1.8100e+00, -1.2019e+00, 1.3268e+00, 9.4743e-01, -3.3756e+00, 4.0646e+00, -4.8882e-01],
        #      [-1.3789e+00, -1.8457e+00, 2.6843e+00, -7.9319e-01, 1.0077e+00, -3.9636e-02, 4.0183e+00, -3.2326e+00, 6.8632e-01, -1.1624e+00],
        #      [-4.9295e-01, -1.5997e+00, 3.1065e+00, 2.6141e+00, -6.5264e-01, 4.6772e-01, -1.2167e+00, -1.6187e+00, 2.1471e+00, -9.6054e-01],
        #      [6.6611e+00, -6.5583e+00, -1.1297e+00, 8.6024e-01, -1.5815e+00, 4.5514e+00, -7.7254e-01, -9.0695e-01, -4.3669e-01, -1.0039e+00],
        #      [8.1183e-01, -4.0013e+00, -2.4532e+00, -1.0732e+00, 1.3184e+00, 7.2588e-01, -2.9442e+00, 4.4685e+00, -3.6961e-01, 3.9793e+00],
        #      [9.5061e-01, -1.4417e+00, 5.0726e-01, 1.1996e-01, -7.5244e-01,  1.1879e+00, 1.9397e-01, -8.8471e-01, 1.1188e+00, -5.5390e-01]],
        #     grad_fn= < AddmmBackward0 >)

        # 모델 예측값과 정답과의 오차(loss)인 손실함수 계산
        # CrossEntropyLoss 손실함수에는 softmax 함수가 포함되어 outputs를 softmax처리하지 않아도 됨
        loss = loss_function(outputs, y_train)
        # print(' loss = ', loss)
        # loss = tensor(2.3035, grad_fn= < NllLossBackward0 >)

        ######################################################################################
        # 역전파 코드. 즉 학습이 진행함에 따라 모델 파라미터(가중치, 바이어스) 업데이트 하면서 최적화 시킴
        ######################################################################################
        # 역전파 단계 전에, optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인) 갱신할
        # 변수들에 대한 모든 변화도(gradient)를 0으로 만듬. 이렇게 하는 이유는 기본적으로
        # .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기 때문
        optimizer.zero_grad()
        # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도 계산
        loss.backward()
        # optimizer의 step 함수를 호출하면 매개변수가 갱신됨
        optimizer.step()

        # 모델에서 계산된 loss 가 있다면, loss.item()을 통해 loss의 스칼라 값을 가져올 수 있음
        # train_loss_sum에 손실함수값을 더함
        train_loss_sum += loss.item()
        # print('train_loss_sum = ', train_loss_sum)
        # print('loss.item() = ', loss.item())
        # # train_loss_sum = 2.303469657897949
        # loss.item() = 2.303469657897949

        # batch size(32) 만큼 train_total에 더함
        train_total += y_train.size(0)  # label 열 사이즈 같음
        # print('train_total = ', train_total)
        # print('y_train.size(0) = ', y_train.size(0))
        # # train_total = 32
        # y_train.size(0) = 32

        train_correct += ((torch.argmax(outputs, 1)==y_train)).sum().item() # 예측한 값과 일치한 값의 합

    # 학습 데이터 평균 오차 계산 = 학습단계 오차 합계 / 전체 train 배치 작업 횟수 (1,594)
    train_avg_loss = train_loss_sum / total_train_batch
    # 학습 데이터 평균 정확도 = 학습 단계 일치건수 / 전체 학습 건수
    train_avg_accuracy = train_correct / train_total * 100
    print('train_total_count = ', train_total)

    # print('for looping count = ', for_looping_count)
    # for looping count =  1594

    return (train_avg_loss, train_avg_accuracy)

################################################################################
# 모델 평가 함수
################################################################################

def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad(): #미분하지 않겠다는 것(모델 하이퍼 파라이며를 업데이터를 시키지 않겠다는 의미)

        val_loss_sum = 0
        val_correct=0
        val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_val = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)  # label 열 사이즈 같음
            val_correct += ((torch.argmax(outputs, 1)==y_val)).sum().item() # 예측한 값과 일치한 값의 합

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = val_correct / val_total * 100

    return (val_avg_loss, val_avg_accuracy)

################################################################################
# 모델 테스트
################################################################################
def model_test(dataloader, model):

    # 신경망을 추론(검증) 단계로 전환
    model.eval()

    with torch.no_grad(): #test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.

        test_loss_sum = 0
        test_correct=0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_test = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_test = labels

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()

            test_total += y_test.size(0)  # label 열 사이즈 같음
            test_correct += ((torch.argmax(outputs, 1)==y_test)).sum().item() # 예측한 값과 일치한 값의 합

        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100*test_correct / test_total

        print('accuracy:', test_avg_accuracy)
        print('loss:', test_avg_loss)


# from datetime import datetime

################################################################################
# 모델 학습 및 평가
################################################################################

# 학습 오차를 리스트로 저장
train_loss_list = []
# 학습 정확도를 리스트로 저장
train_accuracy_list = []

# 검증 오차를 리스트로 저장
val_loss_list = []
# 검증 정확도를 리스트로 저장
val_accuracy_list = []

start_time = datetime.datetime.now()

EPOCHS = 20

for epoch in range(EPOCHS):

    #==============  model train  ================
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)  # training

    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)
    #=============================================

    #============  model evaluation  ==============
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function, optimizer)  # evaluation

    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)

    #============  model evaluation  ==============

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    print(nowDatetime, 'epoch:', '%03d' % (epoch + 1),
          'train loss =', '{:.4f}'.format(train_avg_loss), 'train accuracy =', '{:.4f}'.format(train_avg_accuracy),
          'validation loss =', '{:.4f}'.format(val_avg_loss), 'validation accuracy =', '{:.4f}'.format(val_avg_accuracy))

end_time = datetime.datetime.now()

print('elapsed time => ', end_time-start_time)

################################################################################
# test dataset 으로 정확도 및 오차 테스트
################################################################################
model_test(test_dataset_loader, model)

################################################################################
# 테스트 결과 시각화
################################################################################

import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.plot(val_loss_list, label='validation loss')

plt.legend()

plt.show()

import matplotlib.pyplot as plt

plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(val_accuracy_list, label='validation accuracy')

plt.legend()

plt.show()

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss_list, label='train')
plt.plot(val_loss_list, label='validation')
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(train_accuracy_list, label='train')
plt.plot(val_accuracy_list, label='validation')
plt.legend()

plt.show()
