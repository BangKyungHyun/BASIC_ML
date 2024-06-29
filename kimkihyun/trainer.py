from copy import deepcopy
import numpy as np

import torch

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    # 다음의 _batchify 함수는 매 에포크마다 SGDStochastic Gradient Descent를
    # 수행하기 위해 shuffling 후, 미니배치를 만드는 과정입니다.
    def _batchify(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    # 이 함수는 한 epoch의 학습을 위한 for 반복문을 구현하였습니다.
    # 함수의 시작부분에서 잊지 않고 train() 함수를 호출해서
    # 모델을 학습 모드로 전환하는 것을 확인할 수 있습니다.
    # 만약 이 라인이 생략된다면 이전 에포크의 validation 과정에서
    # 추론 모드였던 모델이 그대로 학습에 활용될 것입니다.
    # for 반복문은 작은 루프를 담당하고,
    # 해당 반복문의 내부는 미니배치의 feed-forward와 back-propagation,
    # 그리고 gradient descent에 의한 파라미터 업데이트가 담겨있습니다.
    # 마지막으로 config.verbose에 따라 현재 학습 현황을 출력합니다.
    # config는 가장 바깥의 train.py에서 사용자의 실행시 파라미터 입력에 따른 설정값이 들어있는 객체입니다.
    # _train 함수의 가장 첫부분에 _batchify 함수를 호출하는 것을 볼 수 있습니다.

    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    # validation 과정에서는 random_split이 필요 없으므로 False로 넘어올 수 있음을 유의하세요.
    # 다음 코드는 검증 과정을 위한 _validate 함수입니다.
    # 대부분 _train 과 비슷하게 구현되어 있음을 알 수 있습니다.
    # 다만 가장 바깥 쪽에 torch.no_grad()가 호출되어 있음을 유의하세요.
    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    # train과 validation 등을 아우르는 큰 loop가 있었고, 학습과 검증 내의 작은 루프가 있었습니다.
    # train 함수 내의 for 반복문은 큰 루프를 구현한 것입니다.
    # 따라서 내부에는 self._train 함수와 self._validate 함수를 호출하는 것을 볼 수 있습니다.
    # 그리고 곧이어 검증 손실 값에 따라 현재까지의 모델을 따로 저장하는 과정도 구현된 것을 확인할 수 있습니다.
    #
    # 현재까지의 최고 성능 모델을 best_model 변수에 저장하기 위해서는
    # state_dict라는 함수를 사용하는 것을 볼 수 있는데요.
    # 이 state_dict 함수는 모델의 가중치 파라미터 값들을 json 형태로 변환하여 리턴합니다.
    # 이 json 값의 메모리를 best_model에 저장하는 것이 아닌,
    # 값 자체를 새로 복사하여 best_model에 할당하는 것을 볼 수 있습니다.
    # 그리고 학습이 종료되면 best_model에 저장된 가중치 파라미터 json 값을
    # load_state_dict를 통해 self.model에 다시 로딩하는 것을 볼 수 있습니다.
    # 이 마지막 라인을 통해서 학습 종료 후에 오버피팅이 되지 않은 가장 좋은 상태의 모델로 복원할 수 있게 됩니다.

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)