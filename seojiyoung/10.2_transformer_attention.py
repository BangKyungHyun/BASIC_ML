#10.2 Transformer attention
#10.2.1 Seq2seq

################################################################################
# 라이브러리 호출
################################################################################

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import sys

import os
# re 모듈은 정규표현식을 사용하고자 할때 이용. 정규표현식은 특정한 규칙을 갖는 문장의 집합을 표현하기 위한 형식
import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 데이터 준비
################################################################################

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

word_count = 0
global word_count1
global word_count2
global word_count3

# 딕셔너리를 위한 클래스
class Lang:

    # Go.Va !
    # Run!    Cours !
    # Run!    Courez !
    # Wow!    Ça
    # alors !

    # 단어의 인덱스를 저장하기 위한 컨테이너를 초기화
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"} # 문장의 시작, 문자의 끝
        self.word2count = {}
        self.n_words = 2                       # SOS와 EOS에 대한 카운트

    # 문장을 단어 단위로 분리한 후 컨테이너(word)에 추가
    def addSentence(self, sentence):

        for word in sentence.split(' '):
            # global word_count
            # word_count += 1
            # if word_count > 50:
            #     sys.exit()
            # print('word =', word)

            # word = go.
            # word = va
            # word = !
            # word = run!
            # word = cours!
            # word = run!
            # word = courez!
            # word = wow!
            # word = ca
            # word = alors!
            # word = fire!
            # word = au
            # word = feu
            # word = !
            # word = help!
            # word = a
            # word = l'aide!
            # word = jump.
            # word = saute.
            # word = stop!
            # word = ca
            # word = suffit!
            # word = stop!
            # word = stop!
            # word = stop!
            # word = arrete - toi
            # word = !
            # word = wait!
            # word = attends
            # word = !
            # word = wait!
            # word = attendez
            # word = !
            # word = i
            # word = see.
            # word = je
            # word = comprends.
            # word = i
            # word =
            # try.
            # word = j
            # 'essaye.
            # word = i
            # word = won!
            # word = j
            # 'ai
            # word = gagne
            # word = !
            # word = i
            # word = won!
            # word = je
            # word = l
            # 'ai
            # word = emporte

            self.addWord(word)

    # 컨테이너에 단어가 없다면 추가되고 있다는 카운트를 업데이트
    def addWord(self, word):
        #
        # global word_count
        # word_count += 1
        # if word_count > 50:
        #     sys.exit()

        if word not in self.word2index:

            self.word2index[word] = self.n_words
            # print('addWord not in => self.word2index[word] =',word,self.n_words,self.word2index, self.word2index[word])
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            # print('addWord     in => self.word2index[word] =',word,self.n_words,self.word2index)

# addWord     in => self.word2index[word] = i 14 {'go.': 2, 'run!': 3, 'wow!': 4, 'fire!': 5, 'help!': 6, 'jump.': 7, 'stop!': 8, 'wait!': 9, 'i': 10, 'see.': 11, 'try.': 12, 'won!': 13}
# addWord     in => self.word2index[word] = won! 14 {'go.': 2, 'run!': 3, 'wow!': 4, 'fire!': 5, 'help!': 6, 'jump.': 7, 'stop!': 8, 'wait!': 9, 'i': 10, 'see.': 11, 'try.': 12, 'won!': 13}
# addWord     in => self.word2index[word] = je 23 {'va': 2, '!': 3, 'cours!': 4, 'courez!': 5, 'ca': 6, 'alors!': 7, 'au': 8, 'feu': 9, 'a': 10, "l'aide!": 11, 'saute.': 12, 'suffit!': 13, 'stop!': 14, 'arrete-toi': 15, 'attends': 16, 'attendez': 17, 'je': 18, 'comprends.': 19, "j'essaye.": 20, "j'ai": 21, 'gagne': 22}
# addWord not in => self.word2index[word] = l'ai 23 {'va': 2, '!': 3, 'cours!': 4, 'courez!': 5, 'ca': 6, 'alors!': 7, 'au': 8, 'feu': 9, 'a': 10, "l'aide!": 11, 'saute.': 12, 'suffit!': 13, 'stop!': 14, 'arrete-toi': 15, 'attends': 16, 'attendez': 17, 'je': 18, 'comprends.': 19, "j'essaye.": 20, "j'ai": 21, 'gagne': 22, "l'ai": 23} 23
# addWord not in => self.word2index[word] = emporte 24 {'va': 2, '!': 3, 'cours!': 4, 'courez!': 5, 'ca': 6, 'alors!': 7, 'au': 8, 'feu': 9, 'a': 10, "l'aide!": 11, 'saute.': 12, 'suffit!': 13, 'stop!': 14, 'arrete-toi': 15, 'attends': 16, 'attendez': 17, 'je': 18, 'comprends.': 19, "j'essaye.": 20, "j'ai": 21, 'gagne': 22, "l'ai": 23, 'emporte': 24} 24

################################################################################
# 데이터 정규화
################################################################################

def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace('[^A-Za-z\s]+', '') # [^A-Za-z\s]+ 등을 제외하고 모두 공백으로 바꿈
    sentence = sentence.str.normalize('NFD') # 유니코드 정규화 방식
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8') # unicode를 ascii로 치환
    return sentence

def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1) # 데이터셋의 첫번째 열 (영어)
    sentence2 = normalizeString(df, lang2) # 데이터셋의 두번째 열 (불어)

    # print('def read_sentence df =\n', df)
    # def read_sentence df =
    #                                                        eng                                                fra
    # 0                                                     Go.                                               Va !
    # 1                                                    Run!                                            Cours !
    # 2                                                    Run!                                           Courez !
    # 3                                                    Wow!                                         Ça alors !
    # 4                                                   Fire!                                           Au feu !
    # ...                                                   ...                                                ...
    # 135837  A carbon footprint is the amount of carbon dio...  Une empreinte carbone est la somme de pollutio...
    # 135838  Death is something that we're often discourage...  La mort est une chose qu'on nous décourage sou...
    # 135839  Since there are usually multiple websites on a...  Puisqu'il y a de multiples sites web sur chaqu...
    # 135840  If someone who doesn't know your background sa...  Si quelqu'un qui ne connaît pas vos antécédent...
    # 135841  It may be impossible to get a completely error...  Il est peut-être impossible d'obtenir un Corpu...
    # [135842 rows x 2 columns]

    # print('def read_sentence lang1 =\n', lang1)

    # def read_sentence lang1 =  eng

    # print('def read_sentence lang2 =\n', lang2)

    # def read_sentence lang2 = fra

    # print('def read_sentence sentence1 =\n', sentence1)
    # def read_sentence sentence1 =
    #  0                                                       go.
    # 1                                                      run!
    # 2                                                      run!
    # 3                                                      wow!
    # 4                                                     fire!
    #                                 ...
    # 135837    a carbon footprint is the amount of carbon dio...
    # 135838    death is something that we're often discourage...
    # 135839    since there are usually multiple websites on a...
    # 135840    if someone who doesn't know your background sa...
    # 135841    it may be impossible to get a completely error...
    # Name: eng, Length: 135842, dtype: object

    # print('def read_sentence sentence2 =\n', sentence2)
    # def read_sentence sentence2 =
    #  0                                                      va !
    # 1                                                    cours!
    # 2                                                   courez!
    # 3                                                 ca alors!
    # 4                                                  au feu !
    #                                 ...
    # 135837    une empreinte carbone est la somme de pollutio...
    # 135838    la mort est une chose qu'on nous decourage sou...
    # 135839    puisqu'il y a de multiples sites web sur chaqu...
    # 135840    si quelqu'un qui ne connait pas vos antecedent...
    # 135841    il est peut-etre impossible d'obtenir un corpu...
    # Name: fra, Length: 135842, dtype: object

    return sentence1, sentence2

################################################################################
# 데이터셋을 불러오기 위해 read_csv 사용
################################################################################

def read_file(loc, lang1, lang2):

    # loc : 예제에서 사용할 데이터셋
    # delimeter : csv파일의 데이터가 어떤 형태(\t,' ','+')로 나뉘었는지 의미
    # header : 일반적으로 데이터셋의 첫 번째를 header(열 이름)로 지정해서 사용되게 되는데
    #          불러올 데이터에 header가 없는 경우에는 header=None 옵션 사용
    # names : 열 이름을 리스트로 형태로 입력. 데이터셋은 총 두개의 열이 있기 때문에 lang1, lang2를 사용
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df

################################################################################
# # 데이터셋 불러오기
################################################################################

def process_data(lang1,lang2):

    df = read_file('../data/%s-%s.txt' % (lang1, lang2), lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang = Lang()
    # print('process_data input_lang =', input_lang)
    # process_data input_lang = <__main__.Lang object at 0x000001DEF86B5310>

    output_lang = Lang()
    # print('process_data output_lang =', output_lang)
    # process_data output_lang = <__main__.Lang object at 0x000001DEF995FE50>

    pairs = []

    # print('process_data len(df) =', len(df))
    # process_data len(df) = 135842

    for i in range(len(df)):

        # MAX_LENGTH = 20
        # split( )을 사용하면 길이에 상관없이 공백을 모두 제거 분리하고, split(' ')을 사용하면 공백 한 개마다 분리하는 것
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            #
            # print('process data sentence1[i] =', sentence1[i])
            # print('process data sentence2[i] =', sentence2[i])
            # print('process data len(sentence1[i].split(' ') =', len(sentence1[i].split(' ')))
            # print('process data len(sentence2[i].split(' ') =', len(sentence2[i].split(' ')))

            # process data sentence1[i] = i won!
            # process data sentence2[i] = je l'ai emporte !
            # process data len(sentence1[i].split() = 2   공백으로 분리하면 2개 단어가 됨
            # process data len(sentence2[i].split() = 4   공백으로 분리하면 4개 단어가 됨

            full = [sentence1[i], sentence2[i]]      # 첫번째와 두번째 열을 합쳐서 저장
            input_lang.addSentence(sentence1[i])     # 입력으로 영어를 사용
            output_lang.addSentence(sentence2[i])    # 출력으로 프랑스어를 사용
            pairs.append(full)                       # pairs에는 입력과 출력이 합쳐진 것을 사용

    return input_lang, output_lang, pairs

################################################################################
# 텐서로 변환
################################################################################
# 문장을 단어로 분리하고 인덱스를 반환
def indexesFromSentence(lang, sentence):

    # print('1. indexesFromSentence lang =',lang)
    # print('2. indexesFromSentence sentence =',sentence)

    # 1. indexesFromSentence lang = <__main__.Lang object at 0x000002615E3D6850>
    # 2. indexesFromSentence sentence = let's leave the decision to our teacher.

    return [lang.word2index[word] for word in sentence.split(' ')]

# 딕셔너리에서 단어에 대한 인덱스를 가져오고 문장 끝에 토큰을 추가
def tensorFromSentence(lang, sentence):

    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    # print('1. tensorFromSentence indexes =',indexes)
    # print('2. tensorFromSentence lang =',lang)
    # print('3. tensorFromSentence sentence =',sentence)

    # 1. tensorFromSentence indexes = [178, 177, 693, 8339, 240, 1290, 1687, 1]
    # 2. tensorFromSentence lang = <__main__.Lang object at 0x000002615E3D6850>
    # 3. tensorFromSentence sentence = let's leave the decision to our teacher.

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# 입력과 출력 문장을 텐서로 변환하여 반환
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    # print('1. tensorsFromPair input_lang = ', input_lang)
    # print('2. tensorsFromPair pair[0] = ', pair[0])
    # print('3. tensorsFromPair input_tensor = ', input_tensor)

    # 1. tensorsFromPair input_lang =  <__main__.Lang object at 0x000002615E3D6850>
    # 2. tensorsFromPair pair[0] =  let's leave the decision to our teacher.
    # 3. tensorsFromPair input_tensor =  tensor([[ 178],
    #         [ 177],
    #         [ 693],
    #         [8339],
    #         [ 240],
    #         [1290],
    #         [1687],
    #         [   1]])

    target_tensor = tensorFromSentence(output_lang, pair[1])
    # print('1. tensorsFromPair output_lang = ', output_lang)
    # print('2. tensorsFromPair pair[1] = ', pair[1])
    # print('3. tensorsFromPair target_tensor = ', target_tensor)

    # 1. tensorsFromPair output_lang =  <__main__.Lang object at 0x000002615E3D60D0>
    # 2. tensorsFromPair pair[1] =  laissons la decision a notre professeur.
    # 3. tensorsFromPair target_tensor =  tensor([[3432],
    #         [ 142],
    #         [5673],
    #         [  10],
    #         [2472],
    #         [3278],
    #         [   1]])

    # global word_count
    # word_count += 1
    # if word_count > 50:
    #     sys.exit()

    return (input_tensor, target_tensor)

################################################################################
# 인코더 네트워크
################################################################################

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim    # 인코더에서 사용할 입력 층
        self.embbed_dim = embbed_dim  # 인코더에서 사용할 입베딩 층
        self.hidden_dim = hidden_dim  # 인코더에서 사용할 은닉 층(이전 은닉층)
        self.num_layers = num_layers  # 인코더에서 사용할 GRU의 계층 갯수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim) # 임베딩 계층 초기화
        # 임베딩 차원, 은닉층 차원, GRU의 계층 갯수를 이용하여 GRU 계층을 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim,num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1) # 임베딩 처리

        # word_count1 = 0
        #
        # word_count1 += 1
        # if word_count1 < 2:
        #     print('Encoder forward src =', src)
        #     print('Encoder forward embedded =', embedded)

        outputs, hidden = self.gru(embedded)          # 임베딩 결과를 GRU 모델에 적용
        return outputs, hidden

################################################################################
# 디코더 네트워크
################################################################################

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim) # 임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim,  # GRU 계층 초기화
                          num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)  # 선형 계층 초기화
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)  # 입력을 (1, 배치크기)로 변경
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))

        # word_count2  = 0
        # word_count2 += 1
        # if word_count2 < 2:
        #     print('Decoder forward input =', input)
        #     print('Decoder forward hidden =', hidden)


        return prediction, hidden

################################################################################
# seq2Seq 네트워크
################################################################################

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        self.encoder = encoder  # 인코더 초기화
        self.decoder = decoder  # 디코더 초기화
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):

        # word_count3 = 0
        #
        # word_count3 += 1
        # if word_count3 < 2:
        #     print('Seq2Seq  =')

        input_length = input_lang.size(0)  # 입력 문자 길이(문장의 단어 수)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim

        # 예측을 출력을 저장하기 위한 변수 초기화
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length): # 문장의 모든 단어를 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])

        # 인코더의 은닉층을 디코더의 은닉층으로 사용
        decoder_hidden = encoder_hidden.to(device)

        # 첫 번째 예측 단어 앞에 토큰(SOS) 추가
        decoder_input = torch.tensor([SOS_token], device=device)

        for t in range(target_length): # 현재 단어에서 출력 단어를 예측
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            outputs[t] = decoder_output

            # teacher_force는 번역(예측)하려는 목표 단어(Ground truth)를 디코더의 다음 입력으로 넣어줌
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (output_lang[t] if teacher_force else topi)

            # teacher force 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token):
                break

        return outputs

################################################################################
# 모델의 오차 계산 함수 정의
################################################################################

teacher_forcing_ratio = 0.5

def Model(model, input_tensor, target_tensor, model_optimizer, criterion):

    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        # 모델의 예측 결과와 정답(예상 결과)를 이용하여 오차를 계산
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss

################################################################################
# 모델 훈련 함수 정의
################################################################################
                                                     # 20000 -> 2000
def trainModel(model, input_lang, output_lang, pairs, num_iteration=2000):

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01) # 옵티마이져 SGD를 사용
    criterion = nn.NLLLoss() # NLLLoss 역시 크로스엔트로피 손실함수와 마찬가지로 분류문제에 사용함
    total_loss_iterations = 0

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 500 == 0:
            average_loss = total_loss_iterations / 500
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))

    torch.save(model.state_dict(), '../data/mytraining.pt')
    return model

################################################################################
# 모델 평가
################################################################################

def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0]) # 입력문자열를 텐서로 변환
        output_tensor = tensorFromSentence(output_lang, sentences[1]) # 출력 문자열을 텐서로 변환
        decoded_words = []
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)  # 각 출력에서 가장 높은 값을 찾아 인덱스로 변환

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>') # EOS 토큰를 만나면 평가를 멈춤
                break
            else: # 예측 결과를 출력 문자열에 추가
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words

# 훈련 데이터셋으로부터 임의의 문장을 가져와서 모델 평가
def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):

    for i in range(n):
        pair = random.choice(pairs)  # 임의의 문장을 가져온다.
        print('input {}'.format(pair[0]))
        print('output {}'.format(pair[1]))
        output_words = evaluate(model, input_lang, output_lang, pair) # 모델 평가 결과는 output_words에 저장
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))

################################################################################
# 모델 훈련 (main module)
################################################################################

lang1 = 'eng'  # 입력으로 사용할 영어
lang2 = 'fra'  # 출력으로 사용할 프랑스어

# 데이터셋 불러오기
input_lang, output_lang, pairs = process_data(lang1, lang2)

# print('input_lang {}' .format(input_lang))
# print('output_lang {}' .format(output_lang))
#
# input_lang <__main__.Lang object at 0x00000283B8115220>
# output_lang <__main__.Lang object at 0x00000283B81150D0>

# print('random sentence {}'.format(randomize))
# random sentence ['tom is married to mary.', 'tom est marie a mary.']

# randomize = pairs
# print('random sentence {}'.format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size)) # 입력과 출력에 대한 단어 수 출력
# Input : 23191 Output : 39387

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 3000 # 30000 -> 3000
# num_iteration = 75000

# 인코더에 훈련 데이터셋을 입력과 모든 출력과 은닉 상태를 저장 (선언만 한 상태)
#                    23191       512           245          1
encoder = Encoder(input_size, hidden_size, embed_size, num_layers)

# 디코더의 첫번째 입력으로 <SOS>토큰이 제공되고 인코더의 마지막 은닉 상태가 디코더의
# 첫번째 은닉상태로 제공 (선언만 한 상태)
#                     39387        512          245          1
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

model = Seq2Seq(encoder, decoder, device).to(device)

print(encoder)
# Encoder(
#   (embedding): Embedding(23191, 256)
#   (gru): GRU(256, 512)
# )
print(decoder)
# Decoder(
#   (embedding): Embedding(39387, 256)
#   (gru): GRU(256, 512)
#   (out): Linear(in_features=512, out_features=39387, bias=True)
#   (softmax): LogSoftmax(dim=1)
# )

model = trainModel(model, input_lang, output_lang, pairs, num_iteration)

################################################################################
# 임의의 문장에 대한 평가 결과
################################################################################
#
print('==================================================')
evaluateRandomly(model, input_lang, output_lang, pairs)

# ################################################################################
# # 어텐션이 적용된 디코더
# ################################################################################
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size) # 임베딩 계층 초기화
#         # 어텐션은 입력을 디코더로 변환. 즉 어텐션은 입력 시퀀스와 길이가 같은 인코딩된 시퀀스로 시퀀스로 변환하는 역할
#         # 따라서 self.max_length는 모든 입력 시퀀스의 최대 길이이여야함
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         # torch.bmm 함수는 배치 행렬 곱(batch matrix multiplication, BMM)을 수행하는 함수
#         # 따라서 attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)은 가중치와
#         # 인코더의 출력 벡터를 곱하겠다는 의미이며, 그 결과(attn_applied)는 입력 시퀀스의 특정 부분에
#         # 관한 정보를 포함하고 있기 때문에 디코더가 적절한 출력 결과를 선택하도록 도와 줌
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
# ################################################################################
# # 어텐션 디코더 모델 학습을 위한 함수
# ################################################################################
#
# def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0
#     plot_loss_total = 0
#     # 인코더와 디코더에 SGD 옵티마이져 적용
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
#                       for i in range(n_iters)]
#     criterion = nn.NLLLoss()
#
#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]  # 입력+출력 쌍에서 입력을 input_tensor로 사용
#         target_tensor = training_pair[1] # 입력+출력 쌍에서 출력을 target_tensor로 사용
#         loss = Model(model, input_tensor, target_tensor, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
#         # 5000 -> 500
#         if iter % 500 == 0: # 모델을 훈련하면서 5000번째마다 오차를 출력
#             print_loss_avg = print_loss_total / 500 # 5000 -> 500
#             print_loss_total = 0
#             print('%d,  %.4f' % (iter, print_loss_avg))
#
# ################################################################################
# # 어텐션 디코더 모델 훈련
# ################################################################################
#
# import time
#
# embed_size = 256
# hidden_size = 512
# num_layers = 1
# input_size = input_lang.n_words
# output_size = output_lang.n_words
#
# encoder1 = Encoder(input_size, hidden_size, embed_size, num_layers)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
#
# print('encoder1 =',encoder1)
# print('attn_decoder1 = ',attn_decoder1)
# # 인코더와 어텐션 디코드를 이용한 모델 생성
# attn_model = trainIters(encoder1, attn_decoder1, 3000, print_every=5000, plot_every=100, learning_rate=0.01)
# # attn_model = trainIters(encoder1, attn_decoder1, 30000, print_every=5000, plot_every=100, learning_rate=0.01)
# # attn_model = trainIters(encoder1, attn_decoder1, 75000, print_every=5000, plot_every=100, learning_rate=0.01)
# print('attn_model = ',attn_model)
