###############################################################################
# 분류기 모델 클래스를 정의
###############################################################################
# 반복되는 형태를 블럭으로 만들어 크기만 다르게 한 후, 필요한 만큼 쌓을 수 있도록 구현할 것임
# 따라서 블럭을 서브 모듈로 넣기 위해 먼저 클래스로 정의

import torch
import torch.nn as nn

###############################################################################
# 하나의 블럭은 nn.Linear 계층, nn.LeakyReLU 활성 함수, nn.BatchNorm1d 계층
# 또는 nn.Dropout 계층 3개로 이루어져 nn.Sequential에 차례대로 선언되어 있음.
# 눈여겨 보아야 할 점은, get_regularizer 함수를 통해
# use_batch_norm이 True이면  nn.BatchNorm1d 계층을 넣어주고,
#                  False이면 nn.Dropout 계층을 넣어준다는 점
# 이렇게 선언된 nn.Sequential은 self.block에 지정되어,
# forward 함수에서 피드포워드 되도록 간단하게 구현됨
#
# 이렇게 선언된 블럭을 모델은 반복해서 재활용 할 수 있음
# 다음의 코드는 최종 모델로써, 앞서 선언된 블럭을 재활용하여 아키텍처를 구성하도록 되어 있음
# 참고로 이 모델은 이후에 작성할 코드에서 MNIST 데이터를 28×28 이 아닌
# 784차원의 벡터로 잘 변환했으리라 가정한 코드임.
# 따라서 추후에 올바른 데이터를 넣어주도록 잊지말고 구현해 주어야 함
###############################################################################

class Block(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(
                dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )

    # forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행
    # 예를 들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로
    # 객체를 호출하면 자동으로 forward 연산이 수행

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)

        return y

###############################################################################
# nn.Sequential을 활용하여 블럭을 필요한 만큼 쌓도록 함.
# 여기서 클래스 선언시에 입력을 받은 hidden_sizes를 통해 필요한 블럭의 갯수와 
# 각 블럭의 입출력 크기를 알 수 있음
# 따라서 hidden_sizes를 활용하여 for 반복문 안에서
# Block 클래스를 선언하여 blocks라는 리스트에 넣어줌. 
# 이렇게 채워진 blocks를 nn.Sequential에 바로 넣어주고, 
# 이어서 각 클래스별 로그 확률 값을 표현하기위한 nn.Linear와 nn.LogSoftmax를 넣어줌. 
# 이후 self.layers에 선언한 nn.Sequential 객체를 넣어주어, 
# forward 함수에서 피드포워드하도록 구현하였음을 볼 수 있음
###############################################################################

class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        print('input_size =', input_size)
        last_hidden_size = input_size
        blocks = []

        for hidden_size in hidden_sizes:

            print('hidden_size =', hidden_size)
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y