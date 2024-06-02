import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

################################################################################
# 1. 데이터 준비
################################################################################

# Load Dataset from sklearn
# 위스콘신 유방암 데이터셋은 30개의 속성을 가지며 이를 통해 유방암 여부 예측
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# print('cancer.DESCR =', cancer.DESCR)

################################################################################
# 판다스로 데이터를 변환
################################################################################
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target
print('df.tail() = \n' ,df.tail())

#       mean radius  mean texture  ...  worst fractal dimension  class
# 564        21.56         22.39  ...                  0.07115      0
# 565        20.13         28.25  ...                  0.06637      0
# 566        16.60         28.08  ...                  0.07820      0
# 567        20.60         29.33  ...                  0.12400      0
# 568         7.76         24.54  ...                  0.07039      1

print('list(df.columns) = ', list(df.columns))

# 총 31 개 속성 - 평균 10개, 표준편차 10개, 최악값 10개, label(class) 1개
# list(df.columns) =
# ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
# 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',

# 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error',
# 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',

# 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
# 'worst concavity','worst concave points', 'worst symmetry', 'worst fractal dimension',
#
# 'class']

# radius : 반지름

print('list(df.columns[:10]) =',list(df.columns[:10]))
# list(df.columns[:10]) =
# ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
# 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
# 'mean fractal dimension']

# 각 속성별 샘플의 점들이 0번 클래스에 해당하는 경우 아래쪽에 1번 클래스에
# 해당하는 경우 위쪽으로 찍혀 있음
# 만약 이 점들의 클래스별 그룹이 특정값을 기준으로 명확하게 나눠진다면 좋다는 것을 확인
# 각 10개 속성이 평균, 표준편차, 최악값을 나타내고 있기 때문에 속성이 30개임

# Pair plot with mean features
# 평균 속성들만 따로 모아서 class속성과 비교하는 페어플롯 표출
sns.pairplot(df[['class'] + list(df.columns[:10])])
# plt.show()

# Pair plot with std features
# 표준편차들만 따로 모아서 class(lable)속성과 비교하는 페어플롯 표출
sns.pairplot(df[['class'] + list(df.columns[10:20])])
# plt.show()

# Pair plot with worst features
# 최악값들만 따로 모아서 class(lable)속성과 비교하는 페어플롯 표출
sns.pairplot(df[['class'] + list(df.columns[20:30])])
# plt.show()

# 특성 속성 11개 선택
cols = ["mean radius", "mean texture",
        "mean smoothness", "mean compactness", "mean concave points",
        "worst radius", "worst texture",
        "worst smoothness", "worst compactness", "worst concave points",
        "class"]

# 0번 클래스는 파란색, 1번 클래스는 주황색으로 표시. 겹치는 영역이 적을수록 좋은 속성임
for c in cols[:-1]:
    sns.histplot(df, x=c, hue=cols[-1], bins=50, stat='probability')
    #plt.show()

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()
print('data.shape = ', data.shape)
# data.shape =  torch.Size([569, 11])

# 선형회귀와 같이 텐서 x와 텐서 y을 가져옴 
x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape =',x.shape, y.shape)
# x.shape, y.shape = torch.Size([569, 10]) torch.Size([569, 1])

################################################################################
# 2. 학습 수행
################################################################################

n_epochs = 28600000
learning_rate = 1e-2
print_interval = 200000

################################################################################
# PyTorch로 신경망 모델을 설계할 때, 크게 다음과 같은 세 가지 스텝을 따르면 된다.
#
# 1. Design your model using class with Variables
# 2. Construct loss and optim
# 3. Train cycle (forward, backward, update)
################################################################################
# 1. Design your model using class with Variables
# ------------------------------------------------------------------------------
# 클래스와 변수를 정의하는 방법을 다루려고 한다.
# PyTorch로 신경망을 설계할 때 크게 두 가지 방법이 있다.
#
# 1. 사용자 정의 nn 모듈 (추천)
# 2. nn.Module을 상속한 클래스 이용
#
# 어느 방식으로 신경망을 설계해도 상관 없지만, 복잡한 신경망을 직접 설계할 때는 내 마음대로
# 정의하는 방식의 nn모듈을 더 많이 사용한다고 한다.
# (참고: tutorials.pytorch.kr/beginner/pytorch_with_examples.html)
# 여기서 모듈이란 한 개 이상의 레이어가 모여서 구성된 것을 말한다. 모듈+모듈 = 새모듈 < 이런 공식도 성립함.
# 신경망(모델)은 한 개 이상의 모듈로 이루어진 내가 최종적으로 원하는 것을 뜻함!
################################################################################
################################################################################
# 1. PyTorch 모델의 기본 구조
################################################################################
# PyTorch로 설계하는 신경망은 기본적으로 다음과 같은 구조를 갖음
# PyTorch 내장 모델 뿐만 아니라 사용자 정의 모델도 반드시 이 구조를 따라야 함
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Model_Name(nn.Module):
#     def __init__(self):
#         super(Model_Name, self).__init__()
#         self.module1 = ...
#         self.module2 = ...
#         """
#         ex)
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#         """
#     def forward(self, x):
#         x = some_function1(x)
#         x = some_function2(x)
#         """
#         ex)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         """
#         return x
# model = Model_Name()  # 여기에 변수를 넣어 주면 됨.
# ------------------------------------------------------------------------------
# PyTorch 모델로 쓰기 위해선 다음 두 가지 조건을 따라야 함. 내장된 모델(nn.Linear등)도 이를 만족함.
#
# 1.torch.nn.Module을 상속해야 함.
#   1) interitance: 상속; 어떤 클래스를 만들 때 다른 클래스의 기능을 그대로 가지고 오는 것임.
# 2. __init()__과 forward()를 override 해야 함
#   1) override: 재정의; torch.nn.Module(부모클래스)에서 정의한 메소드를 자식클래스에서 변경하는 것.
#   2) __init()__에서는 모델에서 사용될 module(nn.Linear, nn.Conv2d), activation function
#     (nn.functional.relu, nn.functional.sigmoid)등을 정의한다.
#   3) forward()에서는 모델에서 실행되어야 하는 계산을 정의한다.
#      backward 계산은 backward()를 이용하면 PyTorch가 알아서 해주니까 forward()만 정의해 주면 된다.
#      input을 넣어서 어떤 계산을 진행하하여 output이 나올지를 정의해 준다고 이해하면 됨.
################################################################################
################################################################################
# nn.Module
################################################################################
# PyTorch의 nn 라이브러리는 Neural Network의 모든 것을 포괄하는 모든 신경망 모델의 Base Class
# 다른 말로, 모든 신경망 모델은 nn.Module의 subclass라고 할 수 있다.
# nn.Module을 상속한 subclass가 신경망 모델로 사용되기 위해선 앞서 소개한 두 메소드를 override 해야 한다.
# 1) __init__(self): initialize; 내가 사용하고 싶은, 내 신경망 모델에 사용될 구성품들을 정의 및 초기화 하는
#    메소드이다.
# 2) forward(self, x): specify the connections;  이닛에서 정의된 구성품들을 연결하는 메소드이다.
################################################################################
# nn.Module을 상속받은 자식 클래스를 정의할 때에는 보통 두개의 함수(메서드)를 오버라이드함.
# 또한 __init__ 함수를 통해 모델을 구성하는데 필요한 내부모듈(선형계층)을 미리 선언함.
# forward함수는 미리 선언된 내부 모듈을 활용하여 계산을 수행함
################################################################################
# 이제 모델을 준비할 차례입니다. 앞서 선형 회귀에서는 선형 계층 하나만 필요했던 것과 달리,
# 이번에는 시그모이드 함수도 모델이 포함되어야 합니다.
# 따라서 nn.Module을 상속받아 클래스를 정의하고, 내부에 필요한 계층들을 소유하도록 합니다.
################################################################################

class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.Sigmoid()    # sigmoid 함수 추가

    def forward(self, x):
        # |x| = (batch_size, input_dim)
        y = self.act(self.linear(x))
        # |y| = (batch_size, output_dim)

        return y

# 로지스틱 회귀 모델 클래스를 생성하고 BCE 손실함수와 옵티마이저 준비
# 모델의 입력 크기는 텐서 X의 마지막 차원 크기가 되고 출력크기는 텐서 Y의 마지막 크기가 됨
model = MyModel(input_dim=x.size(-1),output_dim=y.size(-1))

crit = nn.BCELoss() # MSE 대신 BCE 함수 선언

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

# 선형회귀와 똑같은 코드로 학습을 진행함

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
print(nowDatetime)

for i in range(n_epochs):

    y_hat = model(x)

    loss = crit(y_hat, y)
    # optimizer.zero_grad() 는 반복 시에 전에 계산했던 기울기를 0 으로 초기화하는 함수이다.
    # 즉 최적화된 모든 torch의 기울기를 0으로 바꾼다.
    # 기울기를 초기화해야 새로운 가중치와 편차에 대해서 새로운 기울기를 구할 수 있기 때문이다.
    optimizer.zero_grad()
    # w와 b에 대한 기울기 계산
    loss.backward()
    # model.paramters()에서 리턴되는 변수들의 기울기에 학습률을 곱해서 빼준 뒤에 업데이트한다.
    optimizer.step()

    if (i + 1) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,'Epoch %d: loss=%.4e' % (i + 1, loss))
# 2023-09-13 06:44:22 Epoch 27600000: loss=7.4783e-02
# 2023-09-13 06:45:12 Epoch 27800000: loss=7.4780e-02
# 2023-09-13 06:46:01 Epoch 28000000: loss=7.4777e-02
# 2023-09-13 06:46:50 Epoch 28200000: loss=7.4774e-02
# 2023-09-13 06:47:39 Epoch 28400000: loss=7.4774e-02
# 2023-09-13 06:48:28 Epoch 28600000: loss=7.4774e-02


# Let's see the result!
# y와 y_hat을 비교하여 정확도 계산
correct_cnt = (y == (y_hat > .5)).sum()
total_cnt = float(y.size(0))

print('Accuracy: %.4f' % (correct_cnt / total_cnt))
# Accuracy: 0.9789

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])

sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()

# Breast cancer wisconsin (diagnostic) dataset
# --------------------------------------------
# **Data Set Characteristics:**
#     :Number of Instances: 569
#     :Number of Attributes: 30 numeric, predictive attributes and the class
#     :Attribute Information:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry
#         - fractal dimension ("coastline approximation" - 1)
#
#         The mean, standard error, and "worst" or largest (mean of the three
#         largest values) of these features were computed for each image,
#         resulting in 30 features.  For instance, field 3 is Mean Radius, field
#         13 is Radius SE, field 23 is Worst Radius.
#
# class:
#     - WDBC - Malignant(악성)
#     - WDBC - Benign(양성)
#
# :Summary Statistics:
# == == == == == == == == == == == == == == == == == == = == == == == == ==
#                                           Min       Max
# == == == == == == == == == == == == == == == == == == = == == == == == ==
# radius(mean):                            6.981     28.11
# texture(mean):                           9.71      39.28
# perimeter(mean):                         43.79     188.5
# area(mean):                              143.5     2501.0
# smoothness(mean):                        0.053     0.163
# compactness(mean):                       0.019     0.345
# concavity(mean):                         0.0       0.427
# concave  points(mean):                   0.0       0.201
# symmetry(mean):                          0.106     0.304
# fractal  dimension(mean):                0.05      0.097
# radius(standard error):                  0.112     2.873
# texture(standard error):                 0.36      4.885
# perimeter(standard error):               0.757     21.98
# area(standard error):                    6.802     542.2
# smoothness(standard error):              0.002     0.031
# compactness(standard error):             0.002     0.135
# concavity(standard  error):              0.0       0.396
# concave points(standard error):          0.0       0.053
# symmetry(standard error):                0.008     0.079
# fractal dimension(standarderror) :       0.001     0.03
# radius(worst):                           7.93      36.04
# texture(worst):                          12.02     49.54
# perimeter(worst):                        50.41     251.2
# area(worst):                             185.2     4254.0
# smoothness(worst):                       0.071     0.223
# compactness(worst):                      0.027     1.058
# concavity(worst):                        0.0       1.252
# concave points(worst):                   0.0       0.291
# symmetry(worst):                         0.156     0.664
# fractal dimension(worst):                0.055     0.208
# == == == == == == == == == == == == == == == == == == = == == == == == ==
#
# :Missing Attribute Values: None
# :Class Distribution: 212 - Malignant, 357 - Benign
# :Creator: Dr.William H.Wolberg, W.Nick Street, Olvi L.Mangasarian
# :Donor: Nick Street
# :Date: November, 1995