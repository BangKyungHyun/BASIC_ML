# 5.4.1 특성 맵 시각화
################################################################################
# 라이브러리 호출
################################################################################
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 설명 가능한 네트워크 생성
################################################################################
# 설명 가능한 모델을 위해 13개의 합성곱층과 두개의 완전 연결층으로 구성된 네트워크를 생성
# 이 때 합성곱층과 완전연결층은 렐루 활성화 함수를 사용

class XAI(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(XAI, self).__init__()
        self.features = nn.Sequential(
            # 1)
            nn.Conv2d(3, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            # inplace=True는 기존의 데이터를 연산의 결과값으로 대체하는 것을 의미
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # 2)
            nn.Conv2d(64, 64, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3)
            nn.Conv2d(64, 128, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 4)
            nn.Conv2d(128, 128, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5)
            nn.Conv2d(128, 256, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 6)
            nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 7)
            nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8)
            nn.Conv2d(256, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 9)
            nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 10)
            nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11)
            nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 12)
            nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # 13)
            nn.Conv2d(512, 512, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return F.log_softmax(x)

################################################################################
# 모델 객체화
################################################################################
model=XAI()           # model 이라는 이름의 객체를 생성
model.to(device)      # model을 장치(cpu or gpu)에 할당
model.eval()          # 테스트 데이터에 대한 모델 평가 용도로 사용

print(model)

################################################################################
# 특성 맵을 확인하기 위한 클래스 정의
################################################################################

class LayerActivations:
    features = []

    def __init__(self, model, layer_num):
        # 파이토치는 매 계층마다 print문을 사용하지 않더라고 hook 기능을 사용하여 각 계층의
        # 활성화 함수 또는 기울기 값을 확인할 수 있음. 따라서 register_forward_hook의
        # 목적은 순전파 중에 각 네트워크 모듈의 입력 및 출력을 가져오는 것임
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.detach().numpy()

    def remove(self):
        self.hook.remove()

################################################################################
# 이미지 호출
################################################################################

img=cv2.imread("data/cat.jpg")
plt.imshow(img)

# cv2.resize는 이미지 크기를 변경할 때 사용
# 첫번째 파라미터 : 변경할 이미지 파일
# 두번째 파라미터 : 변경될 이미지 크기를 (너비, 높이) 로 지정
# 세번째 파라미터 : interpolation (보간법)
img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
# ToTensor()(img).unsqueeze(0)에서 사용된 unsqueeze() 는 1차원 데이터를 생성하는 함수
# 즉 이미지 데이터를 텐서로 변환하고 그 변환된 데이터를 1차원으로 변경함
img = ToTensor()(img).unsqueeze(0)
print(img.shape)

################################################################################
# (0) :Conv2d 특성 맵 확인
################################################################################
# 0번째 Conv2d 특성 맵 확인
result = LayerActivations(model.features, 0)

model(img)
activations = result.features

################################################################################
# 특성 맵 확인
################################################################################

fig, axes = plt.subplots(4,4)
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for row in range(4):
    for column in range(4):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*10+column])
plt.show()

################################################################################
# (20) :Conv2d 특성 맵 확인
################################################################################
# 20번째 Conv2d 특성 맵 확인

result = LayerActivations(model.features, 20)

model(img)
activations = result.features

fig, axes = plt.subplots(4,4)
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for row in range(4):
    for column in range(4):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*10+column])
plt.show()

################################################################################
# (40) :Conv2d 특성 맵 확인
################################################################################
# 40번째 Conv2d 특성 맵 확인

result = LayerActivations(model.features, 40)

model(img)
activations = result.features

fig, axes = plt.subplots(4,4)
fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for row in range(4):
    for column in range(4):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*10+column])
plt.show()