# python train.py --model_fn tmp.pth --gpu_id -1 --batch_size 256 --n_epochs 20 --n_layers 5
# python train.py --model_fn ./models/model.pth --n_layers 10 --dropout 0.3
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

###############################################################################
# 파일명	    설명
###############################################################################
# model.py	    : 모델 클래스가 정의된 코드
# trainer.py	: 데이터를 받아와 모델 객체를 학습하기 위한 trainer가 정의된 코드
#               : data loader로부터 준비된 데이터를 넘겨받아 모델에 넣어 학습과 검증을 진행하는 역할을 수행
#               : 학습이 완료되면 모델의 weight parameter는 보통 pickle 형태로 다른 필요한 정보
#                 (e.g. 모델을 생성하기 위한 각종 설정 및 하이퍼파라미터)들과 함께 파일로 저장됨
# utils.py    	: 데이터 파일을 읽어와 전처리를 수행하고, 신경망에 넣기 좋은 형태로 변환하는 코드
# train.py	    : 사용자로부터 하이퍼파라미터를 입력 받아, 필요한 객체들을 준비하여 학습을 진행
#               : 사용자가 학습을 진행할 때 직접 실행(entry point)할 파이썬 스크립트 파일입
# predict.py	: 사용자로부터 기학습된 모델과 추론을 위한 샘플을 입력받아, 추론을 수행
#               : 저장된 피클 파일을 읽어와서 모델 객체를 생성하고, 학습된 가중치 파라미터를 그대로 복원함
#               : 사용자로부터 추론을 위한 샘플이 주어지면 모델에 통과시켜 추론 결과를 반환함. 
#                : 이때, predict.py에 선언된 함수들을 감싸서 RESTful API 서버로 구현할 수도 있을 것임
###############################################################################
# 먼저 define_argparser라는 함수를 통해 사용자가 입력한 파라미터들을 config라는 객체에 저장합니다.
# 다음 코드는 define_argparser 함수를 정의한 코드입니다.
###############################################################################
# argparse 라이브러리를 통해 다양한 입력 파라미터들을 손쉽게 정의하고 처리할 수 있습니다.
# train.py와 함께 주어질 수 있는 입력들은 다음과 같습니다.

# 파라미터 이름	설명	                                        디폴트 값
# model_fn    	모델 가중치가 저장될 파일 경로	                없음. 사용자 입력 필수
# gpu_id	    학습이 수행될 그래픽카드 인덱스 번호 (0부터 시작)	0 또는 그래픽 부재시 -1
# train_ratio	학습 데이터 내에서 검증 데이터가 차지할 비율	    0.8
# batch_size	미니배치 크기                              	256
# n_epochs	    에포크 갯수	                                20
# n_layers	    모델의 계층 갯수	                            5
# use_dropout	드랍아웃 사용여부	                            False
# dropout_p	    드랍아웃 사용시 드랍 확률                   	0.3
# verbose   	학습시 로그 출력의 정도	                        1

# model_fn 파라미터는 required=True 가 되어 있으므로 실행시 필수적으로 입력되어야 합니다.
# 이외에는 디폴트 값이 정해져 있으므로, 사용자가 따로 지정해주지 않으면 디폴트 값이 적용됩니다.
# 만약 다른 알고리즘의 도입으로 이외에도 추가적인 하이퍼파라미터의 설정이 필요하다면
# add_argument 함수를 통해 프로그램이 입력을 받도록 설정할 수 있습니다.
# 이렇게 입력받은 파라미터들은 다음과 같이 접근할 수 있습니다.
###############################################################################

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

###############################################################################
# 앞서 모델 클래스를 정의할 때, hidden_sizes라는 리스트를 통해 쌓을 블럭들의
# 크기들을 지정할 수 있었습니다.
# 사용자가 블럭 크기들을 일일히 지정하는것은 어쩌면 번거로운 일이 될 수 있기 때문에,
# 사용자가 모델의 계층 갯수만 정해주면 자동으로 등차수열을 적용하여
# hidden_sizes를 구하고자 합니다.
# 다음의 get_hidden_sizes 함수는 해당 작업을 수행합니다.
###############################################################################

def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    print('input size = ', input_size)
    print('output size = ', output_size)
    print('n_layers = ', n_layers)
    print('step size int((input_size - output_size) / n_layers) = ', step_size)

    hidden_sizes = []
    current_size = input_size
    print('current_size = ', current_size)

    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]
        print('i = ', i)
        print('hidden_sizes = ', hidden_sizes)
        print('current_size = ', current_size)

    return hidden_sizes

###############################################################################
# 이제 학습에 필요한 대부분의 부분들이 구현되었습니다.
# 이것들을 모아서 학습을 진행하도록 코드를 구현하면 됩니다.
# 다음의 코드는 앞서 구현한 코드들을 모아서 실제 학습을 진행하는 과정을 수행하도록 구현한 코드입니다.

# MNIST에 특화된 입출력 크기를 갖는 것이 아닌,
# 벡터 형태의 어떤 데이터도 입력을 받아 분류할 수 있도록
# input_size와 output_size 변수를 계산하는 것에 주목하세요.
# MNIST에 특화된 하드코딩을 제거하였기 때문에, load_mnist 함수가 아닌
# 다른 로딩 함수로 바꿔치기 한다면 이 코드는 얼마든지 바로 동작할 수 있습니다.
# 사용자로부터 입력받은 configuration을 활용하여 모델을 선언한 이후에,
# Adam 옵티마이저와 NLL 손실 함수도 함께 준비합니다.
# 그리고 트레이너를 초기화한 후, train함수를 호출하여 불러온 데이터를
# 넣어주어 학습을 시작합니다.
# 학습이 종료된 이후에는 torch.save 함수를 활용하여 모델 가중치를 config.model_fn 경로에 저장합니다.
###############################################################################

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    print("input_size = ", input_size)
    print("output_size = ", output_size)

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)

###############################################################################
# 사용자는 train.py를 통해 다양한 파라미터를 시도하고 모델을 학습할 수 있습니다.
# CLI 환경에서 바로 train.py를 호출 할 것이며, 그럼 train.py의 다음 코드가 실행될 것입니다.
###############################################################################

if __name__ == '__main__':
    config = define_argparser()
    main(config)