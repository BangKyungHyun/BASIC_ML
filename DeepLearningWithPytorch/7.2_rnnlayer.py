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

# print(vars(train_data.examples[0]))
# {'text': ["it's", 'been', 'about', '14', 'years', 'since', 'sharon', 'stone', 'awarded', 'viewers', 'a', 'leg-crossing', 'that', 'twisted', 'many', "people's", 'minds.', 'and', 'now,', 'god', 'knows', 'why,', "she's", 'in', 'the', 'game', 'again.', '"basic', 'instinct', '2"', 'is', 'the', 'sequel', 'to', 'the', 'smash-hit', 'erotica', '"basic', 'instinct"', 'featuring', 'a', 'sexy', 'stone', 'and', 'a', 'vulnerable', 'michael', 'douglas.', 'however,', 'fans', 'of', 'the', 'original', 'might', 'not', 'even', 'get', 'close', 'to', 'this', 'one,', 'since', '"instinct', '2"', 'is', 'painful', 'film-making,', 'as', 'the', 'mediocre', 'director', 'michael', 'caton-jones', 'assassinates', 'the', 'legacy', 'of', 'the', 'first', 'film.<br', '/><br', '/>the', 'plot', 'of', 'the', 'movie', 'starts', 'when', 'a', 'car', 'explosion', 'breaks', 'in', 'right', 'at', 'the', 'beginning.', 'catherine', 'tramell', '(sharon', 'stone,', 'trying', 'to', 'look', 'forcefully', 'sexy)', 'is', 'a', 'suspect', 'and', 'appears', 'to', 'be', 'involved', 'in', 'the', 'murder.', 'a', 'psychiatrist', '(a', 'horrible', 'david', 'morrisey)', 'is', 'appointed', 'to', 'examine', 'her,', 'but', 'eventually', 'falls', 'for', 'an', 'intimate', 'game', 'of', 'seduction.<br', '/><br', '/>and', 'there', 'it', 'is,', 'without', 'no', 'further', 'explanations,', 'the', 'basic', 'force', 'that', 'moves', 'this', '"instinct".', 'nothing', 'much', 'is', 'explained', 'and', 'we', 'have', 'to', 'sit', 'through', 'a', 'sleazy,', 'c-class', 'erotic', 'film.', 'sharon', 'stone', 'stars', 'in', 'her', 'first', 'role', 'where', 'she', 'is', 'most', 'of', 'the', 'time', 'a', 'turn-off.', 'part', 'of', 'it', 'because', 'of', 'the', 'amateurish', 'writing,', 'the', 'careless', 'direction,', 'and', 'terrifyingly', 'low', 'chemistry.', 'the', 'movie', 'is', 'full', 'of', 'vulgar', 'dialogues', 'and', 'even', 'more', 'sexuality', '(a', 'menage', 'a', 'trois', 'scene', 'was', 'cut', 'off', 'so', 'that', 'this', "wouldn't", 'be', 'rated', 'nc-17)', 'than', 'the', 'first', 'entrance', 'in', 'the', 'series.', '"instinct"', 'is', 'a', 'compelling', 'torture.<br', '/><br', '/>to', 'top', 'it', 'off,', 'everything', 'that', 'made', 'the', 'original', 'film', 'a', 'guilty', 'pleasure', 'is', 'not', 'found', 'anywhere', 'in', 'the', 'film.', 'the', 'acting', 'here', 'is', 'really', 'bad.', 'sharon', 'stone', 'has', 'some', 'highlights,', 'but', 'here,', 'she', 'gets', 'extremely', 'obnoxious.', 'david', 'morrisey', 'stars', 'in', 'the', 'worst', 'role', 'of', 'his', 'life,', 'and', 'seems', 'to', 'never', 'make', 'more', 'than', 'two', 'expressions', 'in', 'the', 'movie-', 'confused', 'and', 'aroused.', '"instinct', '2"', 'is', 'a', 'horrible', 'way', 'to', 'continue', 'an', 'otherwise', 'original', 'series,', 'that', 'managed', 'to', 'put', 'in', 'thriller', 'with', 'erotica', 'extremely', 'well.', 'paul', 'verhoeven,', 'how', 'i', 'miss', 'you....<br', '/><br', '/>"basic', 'instinct', '2"', 'never', 'sounded', 'like', 'a', 'good', 'movie,', 'and,', 'indeed,', 'it', "isn't.", 'some', 'films', 'should', 'never', 'get', 'out', 'of', 'paper,', 'and', 'that', 'is', 'the', 'feeling', 'you', 'get', 'after', 'watching', 'this.', 'now,', 'it', 'is', 'much', 'easier', 'to', 'understand', 'why', 'douglas', 'and', 'david', 'cronenberg', 'dropped', 'out,', 'and', 'why', 'sharon', 'stone', 'was', 'expecting', 'a', 'huge', 'paycheck', 'for', 'this......-----3/10'], 'label': 'neg'}

# print(train_data.examples[0])
# <torchtext.legacy.data.example.Example object at 0x0000024CF7D63910>

# print(vars(train_data.examples))
# TypeError: vars() argument must have __dict__ attribute

TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

BATCH_SIZE = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# 데이터셋 분리
################################################################################

# BucketIterator을 이용하여 훈련, 검증, 테스트 데이터셋으로 분리

train_iterator, valid_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)

################################################################################
# 변수 값 지정
################################################################################

vocab_size = len(TEXT.vocab)
print('TEXT.vocab = ', TEXT.vocab)
# TEXT.vocab =  <torchtext.legacy.vocab.Vocab object at 0x0000018F8C91E8E0>

print('len(TEXT.vocab) = ', len(TEXT.vocab))
# len(TEXT.vocab) =  10002

print(TEXT.vocab.stoi)
# defaultdict(<bound method Vocab._default_unk_index of <torchtext.legacy.vocab.Vocab object at 0x000002486EA8E910>>, {'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4, 'of': 5, 'to': 6,
# 'is': 7, 'in': 8, 'i': 9, 'this': 10, 'that': 11, 'it': 12, '/><br': 13, 'was': 14, 'as': 15, 'for': 16, 'with': 17, 'but': 18, 'on': 19, 'movie': 20, 'his': 21, 'are': 22, 'not': 23, 'film':
# 24, 'you': 25, 'have': 26, 'he': 27, 'be': 28, 'at': 29, 'one': 30, 'by': 31, 'an': 32, 'they': 33, 'from': 34, 'all': 35, 'who': 36, 'like': 37, 'so': 38, 'just': 39, 'or': 40, 'has': 41,
# 'her': 42, 'about': 43, "it's":
# -----------
# 9972, 'relates': 9973, 'revolution,': 9974, 'rhyme': 9975, 'ride,': 9976, 'riff': 9977, 'rivers': 9978, 'road.': 9979, 'rookie': 9980, 'sake,': 9981, 'sale': 9982,
# 'sarandon': 9983, 'scale.': 9984, 'scheming': 9985, 'secure': 9986, 'senator': 9987, 'serials': 9988, 'seth': 9989, 'sexuality,': 9990, 'shop,': 9991, 'sight.': 9992, 'smile.': 9993,
# 'snowman': 9994, 'so-so': 9995, 'soul,': 9996, 'spade': 9997, 'span': 9998, 'speaking,': 9999, 'spectacle': 10000, 'spectacular.': 10001})
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
        # 크기가(계층의 개수, 배치 크기, 은닉칭의 뉴런/유닛 개수)인 은닉상태(텐서)를 생성하여 0으로 초기화 한 반환

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