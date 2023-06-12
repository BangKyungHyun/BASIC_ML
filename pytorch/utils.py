import torch

# x와 y에는 이미지 데이터와 이에 따른 클래스 레이블이 담겨있을 것입니다.
# 다만, x의 경우에는 원래 28 × 28 이므로 flatten이 True인 경우에는
# view 함수를 통해 784차원의 벡터로 바꿔주는 것을 볼 수 있습니다.
# 또한, 원래 각 픽셀은 0에서 255까지의 그레이 스케일 데이터이기 때문에,
# 이를 255로 나누어서 0에서 1사이의 데이터로 바꿔주는 부분도 눈여겨 봐 주세요.

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y

 # MNIST는 본디 60,000장의 학습 데이터와 10,000장의 테스트 데이터로 나뉘어 있습니다.
#  따라서 우리는 60,000장의 학습 데이터를 다시 학습 데이터와 검증 데이터로
#  나누는 작업을 수행해야 합니다. 다음의 함수는 해당 작업을 수행합니다.
def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y

def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes