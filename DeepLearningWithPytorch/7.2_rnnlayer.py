################################################################################
# 라이브러리 호출
################################################################################
import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

################################################################################
# 데이터셋 내려받기 및 전 처리
################################################################################
start=time.time()

TEXT = torchtext.legacy.data.Field(sequential = True, batch_first = True, lower = True)
LABEL = torchtext.legacy.data.Field(sequential = False, batch_first = True)

from torchtext.legacy import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(split_ratio = 0.8)

TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

BATCH_SIZE = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# 데이터셋 분리
################################################################################

# BucketIterator을 이용하여 훈련, 검증, 테스트 데이터셋으로 분리

train_iterator, valid_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

################################################################################
# 변수 값 지정
################################################################################

vocab_size = len(TEXT.vocab)

# POS(긍정) NEG(부정)
n_classes = 2

################################################################################
# RNN계층 네트워크
################################################################################

class BasicRNN(nn.Module):

    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicRNN, self).__init__()

        self.n_layers = n_layers                      # RNN 계층에 대한 개수
        self.embed = nn.Embedding(n_vocab, embed_dim) # 워드 임베딩 적용
        self.hidden_dim = hidden_dim                  # 드롭아웃 적용
        self.dropout = nn.Dropout(dropout_p)

        # RNN 계층에 대한 문법
        # embed_dim : 훈련 데이터셋의 특성 갯수 (컬럼 갯수)
        # self.hidden_dim : 은닉 계층의 뉴런(유닛) 갯수
        # num_layers : RNN 계층의 갯수
        # batch_first : 기본값은 False로 입력 데이터는(시퀀스의 길이,배치 크기, 특성 갯수) 입니다.
        #               하지만 true로 설정하면 배치 크기가 가장 앞으로 오면서 (배치크기,
        #               시퀀스의 길이, 특성 갯수) 형태가 됨
        self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        # 최초 은닉 상태의 값을 0으로 초기화
        h_0 = self._init_state(batch_size = x.size(0))
            # RNN 계층을 의미하며, 파라미터로 입력과 이전 은닉 상태의 값을 받음
        x, _ = self.rnn(x, h_0)
        # 모든 네트워크를 거쳐서 가장 마지막에 나온 단어의 임베딩 값(마지막 은닉 상태의 값)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit

    def _init_state(self, batch_size = 1):
        # 모델의 파라미터 값을 가져와서 weight 변수에 저장
        weight = next(self.parameters()).data
        # 크기가(계층의 개수, 배치 크기, 은닉칭의 뉴런/유닛 개수)인 은닉상태(텐서)를 생성하여
        # 0으로 초기화 한 반환
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

################################################################################
# 손실함수와 옵티마이저 지정
################################################################################

model = BasicRNN(n_layers = 1, hidden_dim = 256, n_vocab = vocab_size, embed_dim = 128, n_classes = n_classes, dropout_p = 0.5)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

################################################################################
# 모델 학습 함수
################################################################################

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        # sub()는 뺄셈에 대한 함수이며 함수명에 '-'이 붙은 것은 inplace 연산을 하겠다는 의미
        # 그리고 앞에서 IMDB의 레이블의 경우 긍정은 2, 부정은 1의 값을 갖음
        # 따라서 y.data에서 1을 뺀다는 것은 레이블 값을 0가 1로 변환하겠다는 의미
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

        if b % 50 == 0:
            print(nowDatetime, "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(e,
                                                                           b * len(x),
                                                                           len(train_iter.dataset),
                                                                           100. * b / len(train_iter),
                                                                           loss.item()))

################################################################################
# 모델 평가 함수
################################################################################

def evaluate(model, val_iter):
    model.eval()
    corrects, total, total_loss = 0, 0, 0

    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction="sum")
        total += y.size(0)
        total_loss += loss.item()
        # 모델의 정확도를 구함
        # max(1)[1] : max(dim=0)[0]은 최대갓을 나타내고 max(dim=0)[1] 은 최대값을 갖는 
        #             데이터의 인덱스를 나타냄
        # view(y.size()) : logit.max(1)[1]의 결과를 y.size()로 크기를 변경
        # data == y.data : 모델의 예측결과(logit.max(1)[1].view(y.size()).data) 가 
        #         레이블(실제값. y.data)과 같은지 확인
        # sum() : 모델의 예측결과와 레이블(실제 값)이 같으면 그 합을 corrects변수에 누적 저장
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy

################################################################################
# 모델 학습 및 평가
################################################################################

BATCH_SIZE = 100
LR = 0.001
EPOCHS = 5
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iterator)
    val_loss, val_accuracy = evaluate(model, valid_iterator)
    print("[EPOCH: %d], Validation Loss: %5.2f | Validation Accuracy: %5.2f" % (e, val_loss, val_accuracy))

test_loss, test_acc = evaluate(model,test_iterator)
print("Test Loss: %5.2f | Test Accuracy: %5.2f" % (test_loss, test_acc))