
import torch
import torch.nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from model import ImageClassifier

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

# 학습을 마치면 가중치 파라미터가 담긴 파일이 torch.save 함수를 활용하여 피클 형태로
# 저장되어 있을 것입니다.
# 그럼 이제 해당 모델 파일을 불러와서 추론 및 평가를 수행하는 코드를 구현해야 합니다.

# 보통은 train.py처럼 predict.py를 만들어서 일반 파이썬 스크립트로 짤 수도 있지만,
# 좀 더 손쉬운 시각화를 위해서 주피터 노트북을 활용하도록 하겠습니다.

# 만약 단순히 추론만 필요한 상황이라면 predict.py를 만들어서 추론 함수를 구현한 후,
# API 서버 등에서 wrapping하는 형태로 구현할 수 있을 것입니다.

# 다음 코드는 torch.load를 활용하여 torch.save로 저장된 파일을 불러오기 위한 코드 입니다.

model_fn = "./model.pth"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load(fn, device):
    d = torch.load(fn, map_location=device)

    return d['model'], d['config']

def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))

# map_location을 통해서 내가 원하는 device로 객체를 로딩하는 것에 주목하세요.
# 만약 map_location을 쓰지 않는다면, 자동으로 앞서 학습에 활용된
# 디바이스로 로딩될 것입니다.
# 같은 컴퓨터라면 크게 상관이 없을 수도 있지만,

# 만약 다른 컴퓨터인데 GPU가 없거나 갯수가 다르다면 문제가 될 수 있습니다.
# 예를 들어 GPU 4개짜리 컴퓨터에서 3번 GPU를 활용해서 학습이 된 파일인데,
# 추론 컴퓨터에는 0번 GPU까지만 있는 상황이라면 문제가 될 것입니다.
#
# 다음 코드는 추론을 직접 수행하는 코드를 test함수로 구현한 모습입니다.
# eval() 함수를 활용하여 잊지않고 모델을 추론 모드로 바꿔주는 모습입니다.
# 또한 torch.no_grad()를 활용하여 효율적으로 텐서 계산 연산하기 위한 모습도 확인할 수 있습니다.

def test(model, x, y, to_be_shown=True):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)

        if to_be_shown:
            plot(x, y_hat)

# 다만 현재 이 코드의 문제점은, 미니배치 단위로 추론을 수행하지 않는다는 점 입니다.
# MNIST와 같이 작은 데이터에 대해서는 크게 문제되지 않을 수도 있지만,
# 만약 테스트셋이 한번에 연산하기에 너무 크다면 OOM(Out of Memory) 에러가 발생할 것입니다.
# 이 부분은 for 반복문을 통해 간단하게 구현할 수 있으니, 독자분들이 개선해보는 것도 좋은 경험이 될 것입니다.
#
# 다음 코드는 앞서 선언한 코드들을 불러와서 실제 추론을 수행하는 코드입니다.

model_dict, train_config = load(model_fn, device)

# Load MNIST test set.
x, y = load_mnist(is_train=False)
x, y = x.to(device), y.to(device)

input_size = int(x.shape[-1])
output_size = int(max(y)) + 1

model = ImageClassifier(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=get_hidden_sizes(input_size,
                                  output_size,
                                  train_config.n_layers),
    use_batch_norm=not train_config.use_dropout,
    dropout_p=train_config.dropout_p,
).to(device)

# load_state_dict는 json 형태의 모델 가중치가 저장된 객체를 실제 모델 객체에 로딩하는
# 함수입니다.
# 앞서 트레이너trainer 코드를 설명할 때에도 사용된 것을 볼 수 있었는데요.
# 중요한 점은 load_state_dict를 사용하기에 앞서,
# ImageClassifer 객체를 먼저 선언하여 model 변수에 할당하는 것을 볼 수 있습니다.
# 즉, 이렇게 생성된 model 객체는 임의로 초기화(random initialized)된 가중치 파라미터 값을
# 가지고 있을텐데요.
# 이것을 load_state_dict 함수를 통해 기존이 학습이 완료된 가중치 파라미터 값으로
# 바꿔치지 하는 것으로 이해할 수 있습니다.

model.load_state_dict(model_dict)

# 마지막에 test 함수에 전체 테스트셋을 넣어주어,
# 전체 테스트셋에 대한 테스트 성능을 확인할 수 있습니다.
# 다음을 보면 10,000장의 테스트셋 이미지들에 대해서 98.37%의 정확도로 분류를 수행하는 것을 볼 수 있습니다.

test(model, x, y, to_be_shown=False)

# 이것은 모델을 거의 튜닝하지 않은 것이기 때문에,
# 아마 검증 데이터셋을 활용하여 하이퍼파라미터 튜닝을 수행한다면 미미하나마
# 성능의 개선을 얻을 수도 있을 것입니다.[1]
# 중요한 점은 앞서 오버피팅에 관해 배운대로, 절대로 테스트셋을 기준으로
# 하이퍼파라미터 튜닝을 수행해선 안된다는 것입니다.
#
# 다음 코드는 실제 시각화를 위해서 일부 샘플에 대해서 추론 및 시각화를 수행하는 코드와 그 결과입니다.

n_test = 20
test(model, x[:n_test], y[:n_test], to_be_shown=True)

# 2개의 샘플에 대해서는 정확도가 100%가 나오고, 시각화 된 결과를 눈으로 확인해 보았을 때도
# 정답을 잘 맞추는 것을 확인할 수 있습니다.
# 단순히 테스트셋에 대해서 추론 및 정확도 계산만 하고 넘어가기보단,
# 이처럼 실제 샘플들을 뜯어보고 눈으로 확인하며 틀린 것들에 대한 분석을 해야 합니다.
# 독자분들도 틀린 샘플에 대해서만 시각화를 수행하도록 코드를 직접 수정하여,
# 틀린 샘플들을 직접 확인하고 모델을 개선하기위해 분석을 수행해보세요.