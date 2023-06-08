import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

# 파이토치 비전에서 제공하는 datasets 패키지에서 MNIST 데이터셋을 불러옵니다.
# 훌륭하게도 datasets 패키지는 MNIST 데이터셋이 경로에 없다면 자동으로 다운로드 받아 “../data” 경로에 저장합니다.
# MNIST 데이터셋은 테스트셋은 따로 제공하기때문에,
# train 인자를 통해서 학습과 테스트 데이터셋을 각각 불러옵니다.
# 이후에 이 학습 데이터셋은 다시 학습 데이터셋과 검증 데이터셋을 나뉘게 됩니다.

train = datasets.MNIST(
    '../data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)
test = datasets.MNIST(
    '../data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

# 데이터 샘플을 시각화할 수 있는 함수를 미리 하나 만들어보려합니다.
# 지금이야 MNIST라는 유명한 데이터셋을 다루기 때문에 딱히 데이터에 대한 분석 없이 넘어가지만,
# 만약 처음보는 데이터셋이라면 다양한 시각화를 통해 데이터셋에 대한 특성을 먼저 파악해야 할 것입니다.

def plot(x):
    img = (np.array(x.detach().cpu(), dtype='float')).reshape(28, 28)

    plt.imshow(img, cmap='gray')
    plt.show()

# 앞서 만든 plot 함수에 학습 데이터셋의 첫 번째 샘플을 집어넣어보겠습니다.

plot(train.data[0])

# MNIST 샘플의 각 픽셀은 0에서부터 255까지의 숫자로 이루어진 그레이스케일로 구성되어 있습니다.
# 따라서 255로 각 픽셀 값을 나눠주면, 0에서 1까지의 값으로 정규화 할 수 있습니다.
# 그리고 현재 우리의 신경망은 선형 계층으로만 이루어질 것이기 때문에, 2D 이미지도 1차원 벡터로 flatten하여 나타내야 합니다.
# 하나의 샘플은 28 × 28 크기의 픽셀들로 이루어져 있습니다.[3]
# 따라서 이 2차원 행렬을 1차원 벡터로 flatten할 경우, 784 크기의 벡터가 될 것입니다