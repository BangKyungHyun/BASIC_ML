# Split into Train / Valid / Test set
# Load Dataset from sklearn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# 캘리포니아 하우징 데이터셋을 불러옴. Target 열에 우리가 예측해야 하는 출력 값들을 넣어 줌
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df["Target"] = california.target
print('df.tail() =\n',df.tail())

# 필요한 파이토치 패키지들을 불러 옴
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 캘리포니아 하우징 데이터셋의 값을 파이토치 Float 텐서로 변환
# 입력 데이터를 x 변수에 슬라이싱하여 할당하고, 출력 데이터를 y 변수에 슬라이싱하여 할당
data = torch.from_numpy(df.values).float()
x = data[:,:-1]
y = data[:,-1:]

# x와 y의 크기를 출력해 보면, 20640의 샘플들로 데이터셋이 구성된 것을 확인
# 입력 변수는 한 샘플당 8개의 값을 가지고 있으며, 출력 변수는 하나의 값(scalar)으로 되어 있음
print('x.size(), y.size() = \n', x.size(), y.size())
print('x.shape, y.shape = \n', x.shape, y.shape)

# 이렇게 준비된 입출력을 이제 임의로 학습, 검증, 테스트 데이터로 나눔
# 60%의 학습 데이터와 20%의 검증 데이터, 20%의 테스트 데이터로 구성하기 위해서 미리 비율을 설정
# Train / Valid / Test ratio
ratios = [.6, .2, .2]

# ratios에 담긴 값들을 활용하여 실제 데이터셋에서 몇 개의 샘플들이 각각 학습, 검증, 테스트셋에
# 할당되어야 하는지 구할 수 있음
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]

# 전체 20,640개의 샘플들 중에 학습 데이터는 12,384개가 되어야 하며,
# 검증 데이터와 테스트 데이터는 각각 4,128개가 되어야 함
print("Train %d / Valid %d / Test %d samples." % (train_cnt, valid_cnt, test_cnt))

################################################################################
# Shuffle before split.
################################################################################
# 이때 중요한 건 앞의 비율로 데이터셋을 나누되, 임의로 샘플들을 선정하여 나누어야 한다는 것임
# 다음 코드를 통해서 데이터셋 random shuffling 후, 나누기를 수행함
# 또 주목해야 할 점은 x와 y에 대해서 각각 랜덤 선택 작업을 수행하는 것이 아닌,
# 쌍으로 짝지어 수행한다는 점
# x와 y를 각각 따로 섞어 둘의 관계가 깨져 버린다면,아무 의미 없는 노이즈로 가득찬 데이터가 될 것임
# 데이터셋을 나누는 코드는 SGD에서 미니배치를 나누는 것과 상당히 유사함
################################################################################

indices = torch.randperm(data.size(0))
x = torch.index_select(x, dim=0, index=indices)
y = torch.index_select(y, dim=0, index=indices)

################################################################################
# Split train, valid and test set with each count.
################################################################################
# 이 작업이 끝나면, 앞서 6:2:2 의 비율대로 학습, 검증, 테스트셋이 나뉜 것을 확인할 수 있음.
# 이제 학습에 들어가면 12,384개의 학습 샘플들을 매 epoch마다 임의로 섞어 미니배치들을
# 구성하여 학습 iteration을 돌게 됩니다.
################################################################################

x = list(x.split(cnts, dim=0))
y = y.split(cnts, dim=0)
for x_i, y_i in zip(x, y):
    print(x_i.size(), y_i.size())

################################################################################
# Preprocessing
################################################################################
# 데이터셋이 정상적으로 나뉜 것을 확인하면, 데이터셋 normalization 작업을 마저 수행
# 이전 실습에서는 데이터셋을 불러오자마자 표준 스케일링을 통해 정규화를 진행
# 이번 실습에서는 이제서야 정규화를 진행함. 왜냐하면 여기서도 데이터셋을 피팅하는
# 작업이 수행되기 때문
#
# 표준 스케일링의 과정을 자세히 들여다보면, 표준 스케일링을 위해서는 데이터셋의
# 각 열에 대해서 평균과 표준편차를 구해야 함

# 이 과정을 통해서 데이터셋의 각 열의 분포를 추정하고, 추정된 평균과 표준편차를 활용하여
# 표준정규분포(standard normal distribution)로 변환함

# 그러므로 만약 검증 데이터 또는 테스트 데이터를 학습 데이터와 합친 상태에서
# 평균과 표준편차를 추정한다면, 정답을 보는 것과 같음.

# 아주 잘 정의된 매우 큰 데이터셋이라면, 임의로 학습 데이터와 검증/테스트 데이터셋을 나눌 때
# 그 분포가 매우 유사하겠지만, 그렇다고 해서 학습 데이터와 테스트 데이터를 합친 상태에서
# 평균과 표준편차를 추정해도 되는 것은 아님

# 따라서 다음 코드를 보면 학습데이터인 x[0] 에 대해서 표준 스케일링을 fitting시키고,
# 이후에 해당 스케일러를 활용하여 학습(x[0]), 검증(x[1]), 테스트(x[2]) 데이터에 대해서
# 정규화를 진행하는 것을 볼 수 있음

# 이처럼 학습 데이터만을 활용하여 정규화를 진행하는 것은 매우 중요하고 실수가 잦은 부분이니 주의

scaler = StandardScaler()
scaler.fit(x[0].numpy()) # You must fit with train data only.

x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()

df = pd.DataFrame(x[0].numpy(), columns=california.feature_names)
print('df.tail() = \n',df.tail())

################################################################################
#Build Model & Optimizer
################################################################################
# nn.Sequential을 활용하여 모델을 선언하고, 아담 옵티마이저에 등록

model = nn.Sequential(
    nn.Linear(x[0].size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y[0].size(-1)),
)

print('model = \n', model)

optimizer = optim.Adam(model.parameters())

################################################################################
# Train
################################################################################
# 모델 학습에 필요한 셋팅 값을 설정

n_epochs = 400000
batch_size = 256
print_interval = 10

print('\nLearning hyper parameter => Epoch = ', format(n_epochs,','),'print_interval = ',format(print_interval,','))
################################################################################
# 우리가 원하는 모델은 가장 낮은 검증 손실 값(validation loss)을 갖는 모델임
#
# 매 epoch마다 학습이 끝날 때, 검증 데이터셋을 똑같이 feed-forwarding하여
# 검증 데이터셋 전체에 대한 평균 손실 값을 구하고, 이전 에포크의 검증 손실 값과
# 비교하는 작업을 수행해야 함

# 만약 현재 에포크의 검증 손실 값이 이전 에포크 까지의 최저(lowest) 검증 손실 값보다
# 더 낮다면, 최저 검증 손실 값은 새롭게 갱신되고, 현재 에포크의 모델은 따로 저장되어야 할 것임

# 현재 에포크의 검증 손실 값이 이전 에포크 까지의 최저 검증 손실 값보다 크다면,
# 그냥 이번 에포크의 모델을 따로 저장할 필요 없이 넘어가면 됨.

# 그렇게 학습이 모두 끝났을 때, 정해진 에포크가 n_epochs 번 진행되는 동안 최저 검증 손실 값을
# 뱉어냈던 모델이 우리가 원하는 잘 일반화(generaliztaion)된 모델이라고 볼 수 있음

# 따라서 학습이 종료되고 나면 최저 검증 손실 값의 모델을 다시 복원하고 사용자에게 반환하면 됨

# 이를 위해서 최저 검증 손실을 추적하기 위한 변수 lowest_loss와 최저 검증 손실 값을
# 뱉어낸 모델을 저장하기 위한 변수 best_model을 미리 생성하는 모습임

# 이때 best_model에 단순히 현재 모델을 저장한다면 얇은 복사(shallow copy)가 수행되어
# 주소값이 저장되므로, 깊은 복사(deep copy)를 통해 값 자체를 복사하여 저장해야 함

#  깊은 복사(deep copy)를 통해 값 자체를 복사하여 저장하기 위해
#  copy 패키지의 deepcopy 함수를 불러옴
################################################################################

from copy import deepcopy
lowest_loss = np.inf
best_model = None

################################################################################
# 학습 조기 종료(early stopping)을 위한 셋팅 값과,가장 낮은 검증손실 값을 찾아내는 에포크를
# 저장하기 위한 변수 lowest_epoch도 선언함
################################################################################

early_stop = 100
lowest_epoch = 0

print('\nearly_stop hyper parameter => early_stop = ', format(early_stop,','))

# 이제 본격 학습을 위한 for 반복문을 수행

# 달라진 점은 바깥 for 반복문의 후반부에 검증 작업을 위한 코드가 추가된 점.
# 따라서 코드가 굉장히 길어짐

# 설명을 위해 같은 for 반복문 안쪽에 있지만, 검증 작업을 위한 코드는 따로 떼어냄.

# 새롭게 추가된 코드는 학습/검증 손실 값 히스토리를 저장하기 위한 train_history와
# valid_history 변수가 추가 되었다는 것 정도임.

train_history, valid_history = [], []

now = datetime.datetime.now()
time1 = now.strftime('%Y-%m-%d %H:%M:%S')
print('Start Time =',time1)

for i in range(n_epochs):

    # Shuffle before mini-batch split.
    # randperm 함수를 통해서 새롭게 섞어줄 데이터셋의 인덱스 순서를 정함
    indices = torch.randperm(x[0].size(0))

    # index_select 함수를 통해서 이 임의의 순서로 섞인 인덱스 순서대로 데이터셋을 섞음
    x_ = torch.index_select(x[0], dim=0, index=indices)
    y_ = torch.index_select(y[0], dim=0, index=indices)
    # |x_| = (total_size, input_dim)
    # |y_| = (total_size, output_dim)

    # split 함수를 활용하여 원하는 배치사이즈로 텐서를 나누어 주면 미니배치를 만드는
    # 작업이 끝남
    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    # |x_[i]| = (batch_size, input_dim)
    # |y_[i]| = (batch_size, output_dim)

    train_loss, valid_loss = 0, 0

    y_hat = []

    # print ('x_, y_', x_, y_)
    # 바캍쪽 for 수행 횟수
    large_for_count = 0

    for x_i, y_i in zip(x_, y_):

        large_for_count += 1
        # 첫번째 에포크에서만 출력 (12,384건을 batch_size 256 나누면 49번 반복)
        # if i == 0:
        #     print('large_for_count %d: ' % (large_for_count))

        # 신경망 모델 결과를 y_hat_i에 할당
        # |x_i| = |x_[i]|
        # |y_i| = |y_[i]|
        y_hat_i = model(x_i)

        # 실제 target(label)과 학습결과간의 차이(비용, 손실)을 계산
        loss = F.mse_loss(y_hat_i, y_i)

        # optimizer.zero_grad() 는 반복 시에 전에 계산했던 기울기를 0 으로 초기화 하는 함수
        # 즉 최적화된 모든 torch의 기울기를 0으로 바꿈
        # 기울기를 초기화 해야 새로운 가중치와 편차에 대해서 새로운 기울기를 구할 수 있기 때문
        optimizer.zero_grad()

        # w와 b에 대한 기울기 계산
        loss.backward()

        # model.paramters()에서 리턴되는 변수들의 기울기에 학습률을 곱해서 빼준 뒤에 업데이트
        optimizer.step()
        
        ################################################################################
        # loss 변수에 담긴 손실 값 텐서를 float type casting을 통해 단순 float 타입으로 변환하여
        # total_loss 변수에 더하는 것을 볼 수 있음.이 부분이 매우 중요함
        ################################################################################
        # 타입캐스팅 이전의 loss 변수는 파이토치 텐서 타입으로 gradient를 가지고 있음

        # 파이토치의 AutoGrad 동작 원리에 의해서 loss 변수가 계산될 때까지 활용된
        # 파이토치 텐서 변수들이 loss 변수에 줄줄이 엮여 있음

        # 따라서 만약 float 타입캐스팅이 없다면 total_loss도 파이토치 텐서가 될 것이고,
        # 이 total_loss 변수는 해당 에포크의 모든 loss 변수를 엮고 있음

        # 결과적으로 total_loss가 메모리에서 없어지지 않는다면 loss 변수와 그에 엮인 텐서 변수들
        # 모두가 아직 참조 중인 상태이므로 파이썬 garbage collector에 의해서 메모리에서 해제되지 않음
        # 즉, memory leak이 발생하게 됨

        # 더욱이 추후 실습에서 처럼 손실 곡선을 그려보기 위해서 total_loss 변수를 따로
        # 또 저장하기라도 한다면 학습이 끝날때까지 학습에 사용된 대부분의 파이토치 텐서 변수들이
        # 메모리에서 해제되지 않는 최악의 상황이 발생할 수도 있음.

        # 그러므로 앞서와 같은 상황에서는 float 타입캐스팅 또는 detach 함수를 통해 AutoGrad를 위해
        # 연결된 그래프를 잘라내는 작업이 필요

        train_loss += float(loss)
    ################################################################################
    # inner end for
    ################################################################################

    train_loss = train_loss / len(x_)

    # 이처럼 학습 데이터셋을 미니배치로 나누어 한바퀴(epoche) 학습하고 나면, 검증 데이터셋을
    # 활용하여 검증 작업을  수행
    # 학습과 달리 검증 작업은 역전파back - propagation를 활용하여 학습을 수행하지 않음
    # 따라서 gradient를 계산할 필요가 없기 때문에, torch.no_grad 함수를 호출하여
    # with 내부에서 검증 작업을 진행함

    # gradient를 계산하기 위한 배후 작업들이 없어지기 때문에 계산 오버헤드가 줄어들어
    # 속도가 빨라지고 메모리 사용량도 줄어듬
    #
    # with 내부를 살펴보겠습니다.split 함수를 써서 미니배치 크기로 나눠주는 것을 볼 수 있음
    # 앞서 이야기한 것처럼 검증 작업은 메모리 사용량이 적기 때문에, 검증 작업을 위한
    # 미니배치 크기는 학습용보다 더 크게 가져가도 됩니다만,
    # 간편함을 위해서 그냥 기존 학습용 미니배치 크기와 같은 크기를 사용함
    # 그리고 학습과 달리 셔플링 작업이 빠진 것을 볼 수 있음
    # 또한 for 반복문 내부에서도 피드포워드만 있고,역전파 관련 코드는 없는 것을 볼 수 있음

    # You need to declare to PYTORCH to stop build the computation graph.

    # 계산 그래프 빌드를 중지하려면 PYTORCH에 선언해야 함

    # grdient 를 계산할 필요가 없기 때문에 torch.no_grad 함수를 호출하여 with 내부에서
    # 검증 작업을 수행

    with torch.no_grad():
        # You don't need to shuffle the validation set.
        # Only split is needed.
        x_ = x[1].split(batch_size, dim=0)
        y_ = y[1].split(batch_size, dim=0)

        valid_loss = 0

        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = F.mse_loss(y_hat_i, y_i)

            valid_loss += loss

            y_hat += [y_hat_i]

    valid_loss = valid_loss / len(x_)

    # Log each loss to plot after training is done.
    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i + 1) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,'Epoch %d: train loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e lowest_epoch=%d' % (
            i + 1,
            train_loss,
            valid_loss,
            lowest_loss,   # 검증 데이터셋에서 가장 낮은 손실 값
            lowest_epoch + 1
        ))
    # Epoch    400: train  loss = 3.3823e-01 valid_loss = 3.7201e-01    lowest_loss = 3.6391e-01
    # Epoch    800: train  loss = 3.2896e-01 valid_loss = 3.4263e-01    lowest_loss = 3.4245e-01
    # Epoch   1200: train  loss = 3.0605e-01 valid_loss = 3.2399e-01    lowest_loss = 3.2185e-01
    # Epoch   1600: train  loss = 2.9349e-01 valid_loss = 3.0849e-01    lowest_loss = 3.0795e-01
    # Epoch   2000: train  loss = 2.9202e-01 valid_loss = 3.0720e-01    lowest_loss = 3.0113e-01
    # Epoch   2400: train  loss = 2.9125e-01 valid_loss = 3.0730e-01    lowest_loss = 3.0110e-01
    # Epoch   2800: train  loss = 2.8883e-01 valid_loss = 3.2578e-01    lowest_loss = 3.0110e-01
    # Epoch   3200: train  loss = 2.9150e-01 valid_loss = 3.1877e-01    lowest_loss = 3.0110e-01

    # 앞서와 같이 학습과 검증 작업이 끝나고 나면, 검증손실 값을 기준으로 모델 저장 여부를 결정
    # 원하는 것은 검증 손실을 낮추는 것임.

    # 기존 최소 손실 값 변수 lowest_loss와 현재 검증 손실 값 valid_loss를 비교하여 최소 손실 값이 갱신될 경우,
    # 현재 에포크의 모델을 저장[1]

    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i

        # 'state_dict()' returns model weights as key-value.
        # 'state_dict()'는 모델 가중치를 키-값으로 반환

        # Take a deep copy, if the valid loss is lowest ever.
        # 유효한 손실이 가장 낮은 경우 깊은 복사를 수행
        best_model = deepcopy(model.state_dict())
        # print("1. epoch = %d: lowest epoch %d: early_stop %d: interval %d: valid_loss=%.4e lowest_loss=%.4e" %
        #       (i + 1,lowest_epoch + 1, early_stop, (i + 1) - lowest_epoch,  valid_loss, lowest_loss))
    else:
        # 정해진 기간(early_stop 변수)동안 최소 검증 손실 값의 갱신이 없을 경우,
        # 학습을 종료
        # 이 조기 종료(early stopping) 파라미터  또한 하이퍼 파라미터임을 인식해야 함
        # print("2. epoch = %d: lowest epoch %d: early_stop %d: interval %d: valid_loss=%.4e lowest_loss=%.4e" %
        #       (i + 1,lowest_epoch + 1, early_stop, (i + 1) - lowest_epoch,  valid_loss, lowest_loss))
        # 최소 검증 값 갱신횟수+조기중단 값이 전체 에포크 횟수보다 적을 때까지 수행
        # 즉, 최종 최소 검증 값 갱신 이후 조기중단 횟수 만큼 갱신이 일어나지 않으면 강제 종료
#        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
        if lowest_epoch + early_stop < i + 1:
             print("최종 최소 검증 값 갱신 이후 %d 번 에포크 동안 최소 검증값 변화가 없어서 종료함" % early_stop)
             print("최종 최소 검증 에포크 값 + 조기 종료 값 = ", str(lowest_epoch + early_stop))
             print("최종 최소 검증 에포크 값 %d " % lowest_epoch)
             print("현재 에포크 값  ", str(i + 1))

            # 최종 최소 검증 값 갱신 이후 10000 번 에포크 동안 최소 검증값 변화가 없음.
             break

################################################################################
# outter for end
################################################################################

print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))
# The best validation loss from epoch 12648: 2.3347e-01

# [1]: state_dict 함수가 현재 모델 파라미터를 key-value 형태로 주소값을 반환하기에,
# 그냥 변수에 state_dict 결과값을 저장할 경우 에포크가 끝날 때마다 best_model에 저장된 값이
# 바뀔 수 있음
# 따라서 deepcopy를 활용하여 현재 모델의 가중치 파라미터를 복사하여 저장

# 모든 작업이 수행되고 나면, for 반복문을 빠져나와 best_model에 저장되어 있던 가중치 파라미터를
# 모델 가중치 파라미터로 복원.
# 그럼 우리는 최소 검증 손실 값을 얻은 모델으로 되돌릴 수 있게 됨

# Load best epoch's model.
model.load_state_dict(best_model)

# 코드를 수행하면 다음과 같이 출력되는 것을 볼 수 있음
# 539번째 에포크에서 최소 검증 손실 값을 얻었음을 알 수 있음

now = datetime.datetime.now()
time2 = now.strftime('%Y-%m-%d %H:%M:%S')
print('End Time =',time2)
################################################################################
# 만약 여러분이 보기에 손실 값이 좀 더 떨어질 여지가 있다면,
# 조기 종료 파라미터를 늘릴 수도 있음(중요).
################################################################################

# Loss History
# 그럼 이제 train_history, valid_history에 쌓인 손실 값들을 그림으로 그려 확인
# 그림을 통해서 확인하면 화면에 프린트된 숫자들 보다 훨씬 쉽게 손실 값 추세를 확인할 수 있음
plot_from = 10

plt.figure(figsize=(20, 10))
plt.grid(True)
plt.title("Train / Valid Loss History")
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:],
)
plt.yscale('log')
plt.show()

################################################################################
# Let's see the result!
################################################################################
# 먼저 가장 눈에 띄는 부분은 대부분의 구간에서 검증 손실 값이 학습 손실 값 보다 낮다는 점임
# 검증 데이터셋은 학습 데이터셋에 비해서 일부에 불과하기 때문에 편향bias이 있을 수 있음.
# 따라서 우연하게 검증 데이터셋이 좀 더 쉽게 구성이 되었다면,
# 학습 데이터셋에 비해 더 낮은 손실 값을 가질 수도 있음
# 만약 이 두 손실 값이 차이가 너무 크게 나지만 않는다면 크게 신경쓰지 않아도 됨
#
# 그리고 검증 손실 값과 학습 손실 값의 차이가 학습 후반부로 갈수록 점점 더 줄어드는 것을 확인할 수 있음
# 모델이 학습 데이터에만 존재하는 특성을 학습하는 과정이라고도 볼 수 있음

# 하지만 어쨌든 검증 손실 값도 천천히 감소하고 있는 상황이므로,
# 온전한 오버피팅에 접어들었다고 볼 수는 없음.[2]

# 독자 분들도 조기 종료 파라미터 등을 바꿔가며
# 좀 더 낮은 검증 손실 값을 얻기 위한 튜닝을 해보시는 것도 좋음
#
################################################################################
# [2]: 조기 종료를 하지 않고 계속 학습시킨다면 오버피팅이 될 것임(샤베인 방법- 김성훈 교수).
################################################################################
#
# 결과 확인

# 테스트 데이터셋에 대해서도 성능을 확인해 보려 함.
# 우리의 최종 목표는 테스트 성능이 좋은 모델을 얻는 것이지만,
# 이 과정에서 학습 데이터셋과 검증 데이터셋만 활용할 수 있었고, 중간 목표는 검증 손실 값을 낮추는 것임

# 이제 이렇게 얻어진 모델이 테스트 데이터셋에 대해서도 여전히 좋은 성능을 유지하는지 확인해보려 함

# 다음의 코드를 보면 검증 작업과 거의 비슷하게 진행되는 것을 볼 수 있음

# torch.no_grad 함수를 활용하여 with 내부에서 gradient 계산 없이 모든 작업이 수행됨

# 또한 미니배치 크기로 split하여 for 반복문을 통해서 피드포워드feed-forward 함

test_loss = 0
y_hat = []

with torch.no_grad():
    x_ = x[2].split(batch_size, dim=0)
    y_ = y[2].split(batch_size, dim=0)

    for x_i, y_i in zip(x_, y_):
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        test_loss += loss # Gradient is already detached.

        y_hat += [y_hat_i]

test_loss = test_loss / len(x_)
y_hat = torch.cat(y_hat, dim=0)

sorted_history = sorted(zip(train_history, valid_history), key=lambda x: x[1])

print("Train loss: %.4e" % sorted_history[0][0])
print("Valid loss: %.4e" % sorted_history[0][1])
print("Test loss: %.4e" % test_loss)

# Train loss: 2.1962e-01
# Valid loss: 2.3347e-01
# Test loss: 2.6383e-01

# sorted 함수를 활용하여 가장 낮은 검증 손실 값과 이에 대응하는 학습 손실 값을 찾아서,
# 테스트 손실 값과 함께 출력
# 최종적으로 이 모델은 0.296 테스트 손실 값이 나오는 것으로 확인됨.
# 만약 여러분에 다른 방법론이나 모델 구조 변경 등을 한다면, 0.296 테스트 손실 값이 나오는
# 모델이 baseline 모델이 될 것임.
# 그럼 새로운 모델은 이 베이스라인을 이겨야 할 것임.[3]

# 이번에는 앞서의 실습들과 마찬가지로 샘플들에 대한 정답과 예측 값을
# 페어플랏(pair plot) 해보도록 하겠음.
# 앞서의 실습들과 다른 점은 테스트셋에 대해서만 그림을 그려본다는 점임

df = pd.DataFrame(torch.cat([y[2], y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])

sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()

# 따라서 정말 정당한 비교를 위해서는 매번 랜덤하게 학습/검증/테스트셋을 나누기보단,
# 아예 테스트셋을 따로 빼두는 것도 좋음