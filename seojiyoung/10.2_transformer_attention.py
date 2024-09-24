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
#################################################################   ###############

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

word_count = 0

################################################################################
# 딕셔너리를 위한 클래스
################################################################################

class Lang:

    # Go.Va !
    # Run!    Cours !
    # Run!    Courez !
    # Wow!    Ça
    # alors !

    # 단어의 인덱스를 저장하기 위한 컨테이너를 초기화

    # print('=====class Lang def __init__ start =========')

    def __init__(self):
        # print('=====class Lang def __init__ body start =========')

        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"} # 문장의 시작, 문자의 끝
        self.word2count = {}
        self.n_words = 2                       # SOS와 EOS에 대한 카운트
        # print('=====class Lang def __init__ body end =========')

    # print('=====class Lang def __init__ end =========')

    # 문장을 단어 단위(스페이스 기준)로 분리한 후 컨테이너(word)에 추가
    def addSentence(self, sentence):

        for word in sentence.split(' '):

            # print('word =', word)

            self.addWord(word)

    # 컨테이너에 단어가 없다면 추가되고 있다는 카운트를 업데이트
    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.n_words  # 초기값이 2부터 시작
            self.word2count[word] = 1
            self.index2word[self.n_words] = word

            # print('def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] ='
            #                             ,word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words])
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = i 2 {'i': 2} {0: 'SOS', 1: 'EOS', 2: 'i'} 2 i
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = made 3 {'i': 2, 'made': 3} {0: 'SOS', 1: 'EOS', 2: 'i', 3: 'made'} 3 made
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = two. 4 {'i': 2, 'made': 3, 'two.': 4} {0: 'SOS', 1: 'EOS', 2: 'i', 3: 'made', 4: 'two.'} 4 two.
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = j'en 2 {"j'en": 2} {0: 'SOS', 1: 'EOS', 2: "j'en"} 2 j'en
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = ai 3 {"j'en": 2, 'ai': 3} {0: 'SOS', 1: 'EOS', 2: "j'en", 3: 'ai'} 3 ai
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = confectionne 4 {"j'en": 2, 'ai': 3, 'confectionne': 4} {0: 'SOS', 1: 'EOS', 2: "j'en", 3: 'ai', 4: 'confectionne'} 4 confectionne
            # def addWord not in => word, self.n_words, self.word2index, self.index2word, self.word2index[word], self.index2word[self.n_words] = deux. 5 {"j'en": 2, 'ai': 3, 'confectionne': 4, 'deux.': 5} {0: 'SOS', 1: 'EOS', 2: "j'en", 3: 'ai', 4: 'confectionne', 5: 'deux.'} 5 deux.
            self.n_words += 1
        else:
            self.word2count[word] += 1  # 단어별 갯수를 계산
            # print('def addWord     in => word, self.n_words, self.word2index, self.word2index[word],self.index2word[self.n_words] ='
            #       ,word, self.n_words, self.word2index, self.word2index[word], self.index2word[self.n_words])

################################################################################
# 데이터 정규화
################################################################################

def normalizeString(df, lang):

    # print('=====def normalizeString(df, lang): start =========')

    # print('df[lang] 000 =', df[lang])
    sentence = df[lang].str.lower()
    # print('sentence 111 =', sentence)
    sentence = sentence.str.replace('[^A-Za-z\s]+', '') # [^A-Za-z\s]+ 등을 제외하고 모두 공백으로 바꿈
    # print('sentence 222 =', sentence)
    sentence = sentence.str.normalize('NFD') # 유니코드 정규화 방식
    # print('sentence 333 =', sentence)
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8') # unicode를 ascii로 치환
    # print('sentence 444 =', sentence)

    # df[lang] 11111 = 0    I made two.
    # Name: eng, dtype: object
    # sentence 111 = 0    i made two.
    # Name: eng, dtype: object
    # sentence 222 = 0    i made two.
    # Name: eng, dtype: object
    # sentence 333 = 0    i made two.
    # Name: eng, dtype: object
    # sentence 444 = 0    i made two.
    # Name: eng, dtype: object

    # df[lang] 11111 = 0    J'en ai confectionné deux.
    # Name: fra, dtype: object
    # sentence 111 = 0    j'en ai confectionné deux.
    # Name: fra, dtype: object
    # sentence 222 = 0    j'en ai confectionné deux.
    # Name: fra, dtype: object
    # sentence 333 = 0    j'en ai confectionné deux.
    # Name: fra, dtype: object
    # sentence 444 = 0    j'en ai confectionne deux.
    # Name: fra, dtype: object

    # print('=====def normalizeString(df, lang): end =========')

    return sentence

################################################################################
# 문장 읽기
################################################################################

def read_sentence(df, lang_kind_1, lang_kind_2):

    # print('=====def read_sentence(df, lang_kind_1, lang_kind_2): start =========')

    # print('def read_sentence df =\n', df)
    # def read_sentence df =
    #             eng                         fra
    # 0  I made two.  J'en ai confectionné deux.

    # print('def read_sentence lang_kind_1 = ', lang_kind_1)
    # def read_sentence lang_kind_1 =  eng

    # print('def read_sentence lang_kind_2 = ', lang_kind_2)
    # def read_sentence lang_kind_2 = fra

    sentence1 = normalizeString(df, lang_kind_1) # 데이터셋의 첫번째 열 (영어)
    sentence2 = normalizeString(df, lang_kind_2) # 데이터셋의 두번째 열 (불어)

    # print('def read_sentence sentence1 =\n', sentence1)

    # def read_sentence sentence1 =
    #  0    i made two.
    # Name: eng, dtype: object

    # print('def read_sentence sentence2 =\n', sentence2)
    # def read_sentence sentence2 =
    #  0    j'en ai confectionne deux.
    # Name: fra, dtype: object

    # print('=====def read_sentence(df, lang_kind_1, lang_kind_2): end =========')

    return sentence1, sentence2

################################################################################
# 데이터셋을 불러오기 위해 read_csv 사용
################################################################################

def read_file(loc, lang_kind_1, lang_kind_2):

    # print('=====read_file(loc, lang_kind_1, lang_kind_2): start =========')

    #-------------------------------------------------------------------------
    # read_csv 사용법
    #-------------------------------------------------------------------------
    # loc : 예제에서 사용할 데이터셋
    # delimeter : csv파일의 데이터가 어떤 형태(\t,' ','+')로 나뉘었는지 의미
    # header : 일반적으로 데이터셋의 첫 번째를 header(열 이름)로 지정해서 사용되게 되는데
    #          불러올 데이터에 header가 없는 경우에는 header=None 옵션 사용
    # names : 열 이름을 리스트로 형태로 입력. 데이터셋은 총 두개의 열이 있기 때문에 lang_kind_1, lang_kind_2를 사용
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang_kind_1, lang_kind_2])

    # print('def read_file(loc, lang_kind_1, lang_kind_2): df =\n ', df)

    # def read_file(loc, lang_kind_1, lang_kind_2): df =
    #              eng                         fra
    # 0  I made two.  J'en ai confectionné deux.

    # print('=====read_file(loc, lang_kind_1, lang_kind_2): end =========')

    return df

################################################################################
# # 데이터셋 불러오기
################################################################################

def process_data(lang_kind_1,lang_kind_2):

    # print('=====process_data(lang_kind_1,lang_kind_2): start =========')

    # print('def process_data lang_kind_1 =', lang_kind_1)
    # print('def process_data lang_kind_2 =', lang_kind_2)

    # def process_data lang_kind_1 = eng
    # def process_data lang_kind_2 = fra

    # lang_kind_1, lang_kind_2 문자를 받아 들여서 [../data/%s-%s1.txt] 문장을 완성시킴
    df = read_file('../data/%s-%s1.txt' % (lang_kind_1, lang_kind_2), lang_kind_1, lang_kind_2)

    # print('def process_data df =\n', df)
    # def process_data df =            '
    #          eng                         fra
    # 0  I made two.  J'en ai confectionné deux.

    # print('def process_data len(df) =', len(df))
    # def process_data len(df) = 1

    sentence1, sentence2 = read_sentence(df, lang_kind_1, lang_kind_2)

    # print('def process_data sentence1 =', sentence1)
    # def process_data sentence1 = 0    i made two.
    # Name: eng, dtype: object

    # print('def process_data sentence2 =', sentence2)
    # def process_data sentence2 = 0    j'en ai confectionne deux.
    # Name: fra, dtype: object

    input_lang = Lang()
    # print('process_data input_lang =', input_lang)
    # process_data input_lang = <__main__.Lang object at 0x000001DEF86B5310>

    output_lang = Lang()
    # print('process_data output_lang =', output_lang)
    # process_data output_lang = <__main__.Lang object at 0x000001DEF995FE50>

    pairs = []

    # print('process_data len(df) =', len(df))
    # process_data len(df) = 1

    for i in range(len(df)):

        # MAX_LENGTH = 20
        # split( )을 사용하면 길이에 상관없이 공백을 모두 제거 분리하고, split(' ')을 사용하면 공백 한 개마다 분리하는 것
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:

            # print('process data sentence1[i] =', sentence1[i])
            # print('process data sentence2[i] =', sentence2[i])
            # print('process data len(sentence1[i].split(' ') =', len(sentence1[i].split(' ')))
            # print('process data len(sentence2[i].split(' ') =', len(sentence2[i].split(' ')))

            # process data sentence1[i] = i made two.
            # process data sentence2[i] = j'en ai confectionne deux.
            # process data len(sentence1[i].split() = 3   공백으로 분리하면 3개 단어가 됨
            # process data len(sentence2[i].split() = 4   공백으로 분리하면 4개 단어가 됨

            full = [sentence1[i], sentence2[i]]      # 첫번째와 두번째 열을 합쳐서 저장
            input_lang.addSentence(sentence1[i])     # 입력으로 영어를 사용
            output_lang.addSentence(sentence2[i])    # 출력으로 프랑스어를 사용
            pairs.append(full)                       # pairs에는 입력과 출력이 합쳐진 것을 사용

            # print('input_lang  =', input_lang)
            # print('output_lang =', output_lang)
            # print('pairs =', pairs)

            # input_lang  = <__main__.Lang object at 0x000002699DEF1340>
            # output_lang = <__main__.Lang object at 0x000002699DEF1370>
            # pairs = [['i made two.', "j'en ai confectionne deux."]]

    # print('=====process_data(lang_kind_1,lang_kind_2): end =========')

    return input_lang, output_lang, pairs
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#  step1) process_data end
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

################################################################################
# 문장을 단어로 분리하고 인덱스를 반환
################################################################################

def indexesFromSentence(lang, sentence):

    # print('=====indexesFromSentence(lang, sentence): start =========')
    #
    # print('1. indexesFromSentence sentence =',sentence)

    # 1. indexesFromSentence sentence = i made two.
    # 1. indexesFromSentence sentence = j'en ai confectionne deux.

    # print('=====indexesFromSentence(lang, sentence): end =========')

    return [lang.word2index[word] for word in sentence.split(' ')]

########################################################################################################################
# 1. 딕셔너리에서 단어에 대한 인덱스를 가져옴 : indexesFromSentence(lang, sentence)
# 2. 문장 끝에 EOS 토큰을 추가 : indexes.append(EOS_token)
# 3. 텐서로 변환 : torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
########################################################################################################################


# 1차는 input_lang과 'i made two.'를 사용
# 2차는  output_lang과 "j'en ai confectionne deux."를 사용
# input_lang  = <__main__.Lang object at 0x000002699DEF1340>
# output_lang = <__main__.Lang object at 0x000002699DEF1370>
# pairs = [['i made two.', "j'en ai confectionne deux."]]

def tensorFromSentence(lang, sentence):

    # print('=====def tensorFromSentence(lang, sentence): start =========')

    # print('1. tensorFromSentence sentence =',sentence)
    # 1. indexesFromSentence sentence = i made two.

    indexes = indexesFromSentence(lang, sentence)
    # print('2. tensorFromSentence indexes =',indexes)
    # 2. tensorFromSentence indexes = [2, 3, 4]

    ###################################################
    # input index [2,3,4]에 EOS index [1]를 추가하여 크기가 4가 됨
    ###################################################

    indexes.append(EOS_token)
    # print('3. tensorFromSentence indexes =',indexes)
    # 3. tensorFromSentence indexes = [2, 3, 4, 1]

    # print('=====def tensorFromSentence(lang, sentence): end =========')

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# torch.tensor는 어떤 data를 tensor로 copy해주는 함수이다.
# torch.tensor data를 넣었을 때,그 data가 tensor가 아니면 torch.Tensor클래스를 적용하여 복사한다.
# 따라서 t=torch.tensor([1,2,3])처럼 data를 꼭 넣어주어야 한다. 그렇지 않으면 copy할 데이터가 없으니 에러가 난다
# 출처: https://amber-chaeeunk.tistory.com/84 [채채씨의 학습 기록:티스토리]

#############################################################################
# 입력과 출력 문장을 텐서로 변환하여 반환
#############################################################################

def tensorsFromPair(input_lang, output_lang, pair):

    # print('###################################################################')
    # print('########def tensorsFromPair(input_lang, output_lang, pair):########')
    # print('###################################################################')
    #
    # print('=====tensorsFromPair(input_lang, output_lang, pair): start =========')
    #
    # print('1. tensorsFromPair input_lang = ', input_lang)
    # print('1-1. input_lang tensorsFromPair pair[0] = ', pair[0])

    # 1. tensorsFromPair input_lang =  <__main__.Lang object at 0x000001A75D890E30>
    # 1-1. input_lang tensorsFromPair pair[0] =  i made two.

    input_tensor = tensorFromSentence(input_lang, pair[0])
    # print('1-2. input_lang tensorsFromPair input_tensor = ', input_tensor)

    # 1-2. input_lang tensorsFromPair input_tensor =
    # tensor([[2],
    #         [3],
    #         [4],
    #         [1]])
    # print('2. tensorsFromPair output_lang = ', output_lang)
    # print('2-1. input_lang  tensorsFromPair pair[1] = ', pair[1])
    # 2. tensorsFromPair output_lang =  <__main__.Lang object at 0x000001A75D890E60>
    # 2-1. input_lang  tensorsFromPair pair[1] =  j'en ai confectionne deux.

    target_tensor = tensorFromSentence(output_lang, pair[1])
    # print('2-2. input_lang  tensorsFromPair target_tensor = ', target_tensor)

    # 2-2. input_lang  tensorsFromPair target_tensor =
    # tensor([[2],
    #         [3],
    #         [4],
    #         [5],
    #         [1]])

    # print('=====tensorsFromPair(input_lang, output_lang, pair): end =========')
    #
    # print('###################################################################')
    # print('########def tensorsFromPair(input_lang, output_lang, pair):########')
    # print('###################################################################\n')
    #
    return (input_tensor, target_tensor)

################################################################################
#  step 2 end
################################################################################
# 인코더 네트워크 (각 단어별로 encoding 반복 => i, meet ... 별로 수행)
# 주어진 입력 문장을 문맥벡터(context vector)로 인코딩
################################################################################
# 인코더는 입력 문장을 각 단어별로 순서대로 인코딩을 하게 되며,문장의 끝을 표시하는 토큰이 붙음
# 또한 인코더는 임베딩 계층과 GRU 계층으로 구성됨.

#  Input  previous_hidden
#     |           |
#     V           |
#     embedding   |
#             |   |
#             V   V
#              GRU
#              | |
#              V V
#        outputs hidden

# 임베딩 계층은 입력에 대한 임베딩 결과가 저장되어 있는 딕셔너리를 조회하는 테이블과 같음
# 이후 GRU 계층과 연결되는데 GRU 계층은 연속하여 들어오는 입력을 계산
# 또한, 이전 계층의 은닉 상태를 계산한 후 망각 게이트와 업데이트 게이트를 갱신함
################################################################################

class Encoder_Network(nn.Module):

    #                    5           512           256          1
    #                    23191       512           256          1
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):

        # print('=====class Encoder_Network(nn.Module): def __init__body start  =========')

        super(Encoder_Network, self).__init__()
        # 입력과 출력에 대한 단어 수 출력(공백은 계산에서 제외하고 SOS, EOS 2개 문자는 계산 포함)
        self.input_dim = input_dim    # 인코더에서 사용할 입력 층 - 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기(23,191) -> 5
        self.embbed_dim = embbed_dim  # 인코더에서 사용할 임베딩 층 - 임베딩 할 벡터의 차원. 사용자가 정해주는 하이퍼파라미터(256)
        self.hidden_dim = hidden_dim  # 인코더에서 사용할 은닉 층(이전 은닉층)(512)
        self.num_layers = num_layers  # 인코더에서 사용할 GRU의 계층 갯수 (1)

        # ****** 입력값을 임베딩 후 GRU 계층을 통과 시킴 ******
        # 임베딩 계층 초기화
        #                                5                256
        #                                23191            256
        # 임베딩을 할 단어들의 개수, 임베딩 할 벡터의 차원
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)

        # print('Encoder self.embedding =', self.embedding)
        # Encoder  self.embedding.weight = Embedding(5, 256)

        # print('Encoder self.embedding.weight =', self.embedding.weight)

        # Encoder self.embedding.weight = Parameter containing:
        # tensor([[-0.1070, -0.1947, -0.0114,  ...,  1.5221, -2.1226, -1.4714],
        #         [ 0.2172, -0.4880,  1.5195,  ...,  0.2550, -0.2288, -0.8428],
        #         [-0.9794, -1.3526,  0.9334,  ...,  0.0383, -0.4121,  0.6605],
        #         ...,
        #         [ 1.3650, -0.8757, -0.7063,  ..., -0.2161,  1.2508, -1.8584],
        #         [-0.4041,  1.4447, -1.2856,  ..., -0.4097, -0.4470, -1.0880],
        #         [-0.3421,  0.2546, -1.7463,  ...,  0.2570, -0.9309,  0.1897]],
        #        requires_grad=True)

        # 임베딩 차원, 은닉층 차원, GRU의 계층 갯수를 이용하여 GRU 계층을 초기화
        #                        256                512                       1
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim,num_layers=self.num_layers)

        # print('=====class Encoder_Network(nn.Module): def __init__body end =========')

    def forward(self, src):

        # print('=====class Encoder_Network(nn.Module): def forward start =========')

        # print('=====class Encoder forward src =', src)
        # Encoder forward src = tensor([1])

        # - reshape(): reshape은 가능하면 input의 view를 반환하고, 안되면 contiguous(인접한)한 tensor로 copy하고 view를 반환한다.
        # - view(): view는 기존의 데이터와 같은 메모리 공간을 공유하며 stride 크기만 변경하여 보여주기만 다르게 한다. 그래서 contigious해야만 동작하며, 아닌 경우 에러가 발생함.

        embedded = self.embedding(src).view(1, 1, -1) # 임베딩 처리

        # print('=====class Encoder forward embedded.shape =', embedded.shape)
        # Encoder forward embedded.shape = torch.Size([1, 1, 256])

        # i            made         two.         [EOS]
        # tensor([2]), tensor([3]), tensor([4]), tensor([1])에 대해서 각각 [1,1,256] 크기의 임베딩 벡터를 생성함

        # Encoder forward embedded =  36*7 +4 = 256
        # tensor([[[ 0.2215,  0.8102, -1.2375, -1.3954, -0.1623, -1.9292,  1.3573,
        #           -1.0219,  0.8392, -0.3229, -0.2732,  0.4185, -0.4613,  0.9092,
        #            0.4946, -0.2815, -0.6067, -0.5605, -0.2284,  1.5095,  0.3018,
        #            0.6056, -1.3617,  2.8244,  1.1949, -0.0785, -1.3487,  1.8003,
        #            0.6219, -0.5357,  0.8245,  2.0702, -0.2189, -0.0467,  0.6272,
        #            1.7901, -0.7208,  1.5146,  0.0939,  0.4331,  0.4723,  0.8610,
        #           -1.2301,  0.8869, -0.0169, -0.2197,  0.0658, -1.9672,  0.0917,
        #           -0.6169, -0.0764, -1.2058,  1.2717, -0.0666, -0.7653,  0.9771,
        #            2.2817, -0.9432, -0.7737,  0.6714, -0.3643, -1.7460, -0.5409,
        #           -0.8999,  1.6545, -0.4321, -0.7969]]], grad_fn=<ViewBackward0>)

        # 임베딩 결과를 GRU 모델에 적용
        # outputs : 현재 단어의 출력 정보 [단어개수, 배치크기, 히든차원]
        # hidden : 현재까지의 모든 단어 정보 [레이어 개수, 배치크기, 히든차원]

        outputs, hidden = self.gru(embedded)

        # print('=====class Encoder forward outputs.shape = ', outputs.shape)
        # hidden이 512개이면 outputs도 512개
        #                                        단어개수, 배치크기, 히든차원
        # Encoder forward outputs.shape =  torch.Size([1, 1, 512])

        # print('=====class Encoder forward hidden.shape = ', hidden.shape)
        #                                        레이어 개수, 배치크기, 히든차원
        # Encoder forward hidden.shape =  torch.Size([1, 1, 512])

        # print('=====class Encoder_Network(nn.Module): encoder_outputs = ', outputs)
        # print('=====class Encoder_Network(nn.Module): encoder_hidden = ', hidden)

        # print('=====class Encoder_Network(nn.Module): def forward end =========')

        # 문맥 벡터 (context vector) 반환
        # outputs : 현재 단어의 출력 정보 [단어개수, 배치크기, 히든차원]
        # hidden : 현재까지의 모든 단어 정보 [레이어 개수, 배치크기, 히든차원]
        return outputs, hidden

################################################################################
# 디코더 네트워크
################################################################################
#  Input  previous_hidden
#     |           |
#     V           |
#     embedding   |
#             |   |
#             V   |
#            ReLU |
#              |  |
#              V  V
#              GRU
#              | |
#              V  |
#          linear  |
#              |    |
#              V     |
#           Softmax   |
#              |       |
#              V       V
#           outputs  hidden

# 입베딩 계층에서는 출력을 위해 딕셔너리를 조회할 테이블 만들며, GRU계층에서는 다음 단어를 예측하기 위한
# 확률을 계산.
# 그 후 선형 계층에서는 계산된 확률 값 중 최적의 값(최종 출력 단어)을 선택하기 위해
# 소프트맥스 활성화 함수를 선정
################################################################################

class Decoder_Network(nn.Module):

    #                     39387        512          256          1
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):

        # print('=====class Decoder_Network(nn.Module): def __init__ body start =========')

        super(Decoder_Network, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # 임베딩 계층 초기화                 39387           256
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        # GRU 계층 초기화            256            512                      1
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim,num_layers=self.num_layers)
        # 선형 계층 초기화               512        39387
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        # print('=====class Decoder_Network(nn.Module): def __init__ body end =========')
    #
    # print('=====class Decoder_Network(nn.Module): def __init__ end =========')

    def forward(self, input, hidden):

        # print('=====class Decoder_Network(nn.Module): def forward start =========')
        # print('=====class Decoder_Network(nn.Module): Decoder forward input =', input)
        # =====class Decoder_Network(nn.Module): Decoder forward input = tensor([0])

        # 입력을 (1, 배치크기)로 변경
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))

        # print('=====class Decoder_Network(nn.Module): Decoder forward output.shape =', output.shape)
        # print('=====class Decoder_Network(nn.Module): Decoder forward hidden.shape =', hidden.shape)
        # print('=====class Decoder_Network(nn.Module): Decoder forward prediction =', prediction)
        #
        # print('=====class Decoder_Network(nn.Module): def forward end =========')

        # prediction : 현재 단어의 출력 정보 [단어개수, 배치크기, 히든차원]
        # hidden : 현재까지의 모든 단어 정보 [레이어 개수, 배치크기, 히든차원]

        return prediction, hidden

################################################################################
# seq2Seq 네트워크
# 인코더는 주어진 소스문장을 context vector로 인코딩합니다.
# 디코더는 주어진 context vector를 타켓 문장으로 디코딩합니다.
# 단, 디코더는 한 단어씩 넣어서  한번씩 결과를 구합니다.
################################################################################

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):

        # print('=====class Seq2Seq(nn.Module): def __init__body start =========')

        super().__init__()

        self.encoder = encoder  # 인코더 초기화
        self.decoder = decoder  # 디코더 초기화
        self.device = device

        # print('=====class Seq2Seq(nn.Module): def __init__body end =========')

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):

        # print('=====class Seq2Seq(nn.Module): def forward start =========')

        input_length = input_lang.size(0)  # 입력 문자 길이(문장의 단어 수)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim

        # print('=====class Seq2Seq(nn.Module): def forward input_lang = ', input_lang)
        # print('=====class Seq2Seq(nn.Module): def forward input_lang.shape[0] = ', input_lang.shape[0])
        # print('=====class Seq2Seq(nn.Module): def forward input_lang.shape[1] = ', input_lang.shape[1])
        # print('=====class Seq2Seq(nn.Module): def forward output_lang = ', output_lang)
        # print('=====class Seq2Seq(nn.Module): def forward output_lang.shape[0] = ', output_lang.shape[0])
        # print('=====class Seq2Seq(nn.Module): def forward output_lang.shape[1] = ', output_lang.shape[1])

        # =====class Seq2Seq(nn.Module): def forward input_lang =  tensor([[2],
        #         [3],
        #         [4],
        #         [1]])
        # =====class Seq2Seq(nn.Module): def forward input_lang.shape[0] =  4
        # =====class Seq2Seq(nn.Module): def forward input_lang.shape[1] =  1
        # =====class Seq2Seq(nn.Module): def forward output_lang =  tensor([[2],
        #         [3],
        #         [4],
        #         [5],
        #         [1]])
        # =====class Seq2Seq(nn.Module): def forward output_lang.shape[0] =  5
        # =====class Seq2Seq(nn.Module): def forward output_lang.shape[1] =  1

        # print('=====class Seq2Seq(nn.Module): def forward input_length = ', input_length)
        # print('=====class Seq2Seq(nn.Module): def forward batch_size = ', batch_size)
        # print('=====class Seq2Seq(nn.Module): def forward target_length = ', target_length)
        # print('=====class Seq2Seq(nn.Module): def forward vocab_size = ', vocab_size)

        # input_length =  4
        # batch_size =  1
        # target_length =  5
        # vocab_size =  6

        # 예측된 출력을 저장하기 위한 변수 초기화
        #                        5                 1        6
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)
        # print('=====class Seq2Seq(nn.Module): def forward outputs = ', outputs)

        # outputs =  \
        #  tensor([[[0., 0., 0., 0., 0., 0.]],
        #          [[0., 0., 0., 0., 0., 0.]],
        #          [[0., 0., 0., 0., 0., 0.]],
        #          [[0., 0., 0., 0., 0., 0.]],
        #          [[0., 0., 0., 0., 0., 0.]]])

        # print('encoder encoder encoder encoder encoder encoder encoder encoder encoder ')
        # print('encoder encoder encoder encoder encoder encoder encoder encoder encoder ')
        # print('encoder encoder encoder encoder encoder encoder encoder encoder encoder ')

        for i in range(input_length):

            # print('=====class Seq2Seq(nn.Module): def forward input_length = ',i+1, input_length)
            # print('=====class Seq2Seq(nn.Module): def forward input_lang[i] =', input_lang[i])
            # =====class Seq2Seq(nn.Module): def forward input_length =  0 4
            # =====class Seq2Seq(nn.Module): def forward input_lang[i] = tensor([2])

            #############################################################
            # 문장의 모든 단어를 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
            #############################################################

        # 인코더의 은닉층을 디코더의 은닉층으로 사용
        decoder_hidden = encoder_hidden.to(device)

        # print('=====class Seq2Seq(nn.Module): def forward decoder_hidden = ', decoder_hidden)

        # 첫 번째 예측 단어 앞에 토큰(SOS) 추가
        decoder_input = torch.tensor([SOS_token], device=device)

        # print('decoder decoder decoder decoder decoder decoder decoder decoder decoder ')
        # print('decoder decoder decoder decoder decoder decoder decoder decoder decoder ')
        # print('decoder decoder decoder decoder decoder decoder decoder decoder decoder ')

        # 타켓 단어의 갯수만큼 반복하여 디코더에 forwarding
        # 현재 단어에서 출력 단어를 예측
        for t in range(target_length):

            # print('=====class Seq2Seq(nn.Module): def forward target_length = ', t, target_length)
            # print('=====class Seq2Seq(nn.Module): def forward decoder_hidden.shape = ', decoder_hidden.shape)
            # print('=====class Seq2Seq(nn.Module): def forward decoder_input.shape = ', decoder_input.shape)
            #
            # # =====class Seq2Seq(nn.Module): def forward target_length =  0 5
            # =====class Seq2Seq(nn.Module): def forward decoder_hidden.shape =  torch.Size([1, 1, 512])
            # =====class Seq2Seq(nn.Module): def forward decoder_input.shape =  tensor([0])

            ####################################################################################################
            ## 중요 : 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타켓 문장을 반환
            ## 중요 :  self.decoder(decoder_input,decoder_hidden) 의 decoder_hidden 가 다시 decoder_hidden으로 사용됨
            ####################################################################################################

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # print('=====class Seq2Seq(nn.Module): def forward decoder_hidden.shape 2 = ', decoder_hidden)

            outputs[t] = decoder_output # FC를 거쳐서 나온 현재의 출력 단어 정보

            # teacher_force는 번역(예측)하려는 목표 단어(Ground truth)를 디코더의 다음 입력으로 넣어줌
            # teacher_forcing_ratio : 학습할 때 실제 목표 단어를 사용하는 비율

            # teacher_force = random.random() < teacher_forcing_ratio 의 코드는 말 그대로 input으로 이용되는 데이터의 조건을 무작위로 선별하는 코드입니다.
            # random.random() 코드는 실행될 때마다 0~1 사이 실수 값을 반환해주며, teacher_forcing_ratio 값이 0.5로 제한하여,
            # 무작위의 값이 1/2의 확률로 낮게 발생되기 때문에, input 으로 이용되는 데이터를 무작위로 다르게 설정할 수 있습니다

            # teacher_force = random_value < teacher_forcing_ratio
            # random_value 가 teacher_forcing_ratio 보다 작으면 teacher_force가 TRUE 아니면 FALSE 가 됨

            random_value = random.random()
            teacher_force = random_value < teacher_forcing_ratio

            # print('teacher_force  = ', teacher_force)
            # print('random_value = ', random_value)
            # print('teacher_forcing_ratio = ', teacher_forcing_ratio)

            #-------------------------------------------------------------------------------
            # torch.topk() 는 input tensor의 가장 높은 k 개의 value값들과 index를 뽑아주는 함수이다.
            #-------------------------------------------------------------------------------
            # >>> x = torch.arange(1., 6.)
            # >>> x
            # tensor([ 1.,  2.,  3.,  4.,  5.])
            # >>> torch.topk(x, 3)
            # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))

            topv, topi = decoder_output.topk(1)
            # print('decoder_output.topk(1) =', decoder_output.topk(1))
            # print('topv =', topv)
            # print('topi =', topi)
            # -1.6180 이 가장 큼(모두 마이너스이여서 숫자가 가장 작은 것이 큼)
            # prediction = tensor([[-1.8532, -1.7986, -1.6180, -1.9043, -1.9382, -1.6791]]
            # decoder_output.topk(1) = torch.return_types.topk(1) values=tensor([[-1.6180]], grad_fn=<TopkBackward0>), indices=tensor([[2]]))
            # topv = tensor([[-1.6180]], grad_fn=<TopkBackward0>)
            # topi = tensor([[2]])

            # teacher_force가 true 이면 목표 단어(output_lang[t])을 다음 입력으로 사용
            #                 false이면 자체 예측 단어를 다음 입력으로 사용

            input = (output_lang[t] if teacher_force else topi)
            # print('input = ', input)
            # print('output_lang[t] = ', output_lang[t])
            # print('teacher_force = ', teacher_force)
            # print('topi = ', topi)

            # teacher_force = True 이여서 input은 목표 단어(output_lang[t] )를 사용
            # input = tensor([3])
            # output_lang[t] = tensor([3])
            # teacher_force = True
            # topi = tensor([[5]])
            #
            # teacher_force = False이여서 input은 topi 를 사용
            # input = tensor([[5]])
            # output_lang[t] = tensor([4])
            # teacher_force = False
            # topi = tensor([[5]])

            # print('teacher_force = ', teacher_force)
            # print('input.item() = ', input.item())
            # print('EOS_token = ', EOS_token)

            # teacher force 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token):
                # print('break break break break break break break break ')
                # print('break break break break break break break break ')
                # print('break break break break break break break break ')

                break

        # print('=====class Seq2Seq(nn.Module): def forward end =========')

        return outputs

################################################################################
# 모델의 오차 계산 함수 정의
################################################################################

teacher_forcing_ratio = 0.5

def Model(Seq2Seq_model, input_tensor, target_tensor, model_optimizer, criterion):

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): start =========')
    #
    # print('=====def Model Seq2Seq_model = ', Seq2Seq_model)
    # print('=====def Model input_tensor = ', input_tensor)
    # print('=====def Model target_tensor = ', target_tensor)
    # print('=====def Model model_optimizer = ', model_optimizer)
    # print('=====def Model criterion = ', criterion)

    # =====def Model Seq2Seq_model =
    # Seq2Seq(
    #   (encoder): Encoder_Network(
    #     (embedding): Embedding(5, 256)
    #     (gru): GRU(256, 512)
    #   )
    #   (decoder): Decoder_Network(
    #     (embedding): Embedding(6, 256)
    #     (gru): GRU(256, 512)
    #     (out): Linear(in_features=512, out_features=6, bias=True)
    #     (softmax): LogSoftmax(dim=1)
    #   )
    # )
    # =====def Model input_tensor =  tensor([[2],
    #         [3],
    #         [4],
    #         [1]])
    # =====def Model target_tensor =  tensor([[2],
    #         [3],
    #         [4],
    #         [5],
    #         [1]])
    # =====def Model model_optimizer =  SGD (
    # Parameter Group 0
    #     dampening: 0
    #     differentiable: False
    #     foreach: None
    #     fused: None
    #     lr: 0.01
    #     maximize: False
    #     momentum: 0
    #     nesterov: False
    #     weight_decay: 0
    # )
    # =====def Model criterion =  NLLLoss()

    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): input length = ',input_length)
    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): input tensor = ',input_tensor)

    # input length = 4
    # input  tensor = tensor([[2],
    #                  [3],
    #                  [4],
    #                  [1]])

    loss = 0
    epoch_loss = 0
    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): model =========\n', Seq2Seq_model)
    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): output = model(input_tensor, target_tensor ) start =========')

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): model(input_tensor, target_tensor ) =========\n', Seq2Seq_model(input_tensor, target_tensor ))

    output = Seq2Seq_model(input_tensor, target_tensor)
    # Seq2Seq(
    #     (encoder): Encoder_Network(
    #     (embedding): Embedding(5, 256)
    # (gru): GRU(256, 512)
    # )
    # (decoder): Decoder_Network(
    #     (embedding): Embedding(6, 256)
    # (gru): GRU(256, 512)
    # (out): Linear(in_features=512, out_features=6, bias=True)
    # (softmax): LogSoftmax(dim=1)
    # )
    # )

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): output = model(input_tensor, target_tensor ) end =========')

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): output = \n', output)
    # output =
    #  tensor([[[-1.7506, -1.8085, -1.7836, -1.8240, -1.7847, -1.8008]],
    #          [[-1.7385, -1.8067, -1.7858, -1.7889, -1.8095, -1.8234]],
    #          [[-1.7328, -1.8109, -1.7883, -1.7667, -1.8121, -1.8436]],
    #          [[-1.7300, -1.8153, -1.7910, -1.7542, -1.8090, -1.8560]],
    #          [[-1.7289, -1.8183, -1.7935, -1.7480, -1.8054, -1.8623]]],
    #         grad_fn=<CopySlices>)

    num_iter = output.size(0)
    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): num_iter = ', num_iter)
    # num_iter =  5

    for ot in range(num_iter):
        # 모델의 예측 결과와 정답(예상 결과)를 이용하여 오차를 계산
        loss += criterion(output[ot], target_tensor[ot])
        # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): output[ot] = \n', output[ot])
        # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): target_tensor[ot] = \n', target_tensor[ot])
        # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): loss = \n', loss)

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion):epoch_loss = ', epoch_loss)

    # epoch_loss = 0.736637020111084

    # print('=====def Model(model, input_tensor, target_tensor, model_optimizer, criterion): end =========')

    return epoch_loss

################################################################################
# 모델 훈련 함수 정의
################################################################################
#
                     # 20000 -> 2000
def trainModel(Seq2Seq_model, input_lang, output_lang, pairs, num_iteration=2000):

    # print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): start =========\n')
    #
    # print('def trainModel -> Seq2Seq_model =', Seq2Seq_model)
    # print('def trainModel -> input_lang =', input_lang)
    # print('def trainModel -> output_lang =', output_lang)
    # print('def trainModel -> pairs =', pairs)

    # def trainModel ->
    # Seq2Seq_model =
    # Seq2Seq(
    #   (encoder): Encoder_Network(
    #     (embedding): Embedding(5, 256)
    #     (gru): GRU(256, 512)
    #   )
    #   (decoder): Decoder_Network(
    #     (embedding): Embedding(6, 256)
    #     (gru): GRU(256, 512)
    #     (out): Linear(in_features=512, out_features=6, bias=True)
    #     (softmax): LogSoftmax(dim=1)
    #   )
    # )
    # def trainModel -> input_lang = <__main__.Lang object at 0x0000026FA24746A0>
    # def trainModel -> output_lang = <__main__.Lang object at 0x0000026FA2474340>
    # def trainModel -> pairs = [['i made two.', "j'en ai confectionne deux."]]

    Seq2Seq_model.train()
    optimizer = optim.SGD(Seq2Seq_model.parameters(), lr=0.01) # 옵티마이져 SGD를 사용
    # criterion : 판단이나 결정을 위한 기준
    criterion = nn.NLLLoss() # NLLLoss 역시 크로스엔트로피 손실함수와 마찬가지로 분류문제에 사용함
    total_loss_iterations = 0

    # random.choice : 지정된 sequence(리스트 등)에서 무작위로 추출하는 함수

    # 입력과 출력 문장을 텐서로 변환하여 반환
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(num_iteration)]

    # print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): training paris = \n', training_pairs )
    # training paris =
    #  [(tensor([[2],
    #         [3],
    #         [4],
    #         [1]]), tensor([[2],
    #         [3],
    #         [4],
    #         [5],
    #         [1]]))]

    for iter in range(1, num_iteration + 1): # range의 결과는 시작숫자부터 종료숫자 바로 앞 숫자까지 컬렉션을 만듭니다.

        print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): for 문 시작 =========')

        print('iter = ', iter)

        training_pair = training_pairs[iter - 1]
        print('training_pairs = ', training_pairs)
        print('training_pair = ', training_pair)
        #
        # training_pairs = [(tensor([[2],
        #                            [3],
        #                            [4],
        #                            [1]]), tensor([[2],
        #                                           [3],
        #                                           [4],
        #                                           [5],
        #                                           [1]]))]
        # training_pair = (tensor([[2],
        #                          [3],
        #                          [4],
        #                          [1]]), tensor([[2],
        #                                         [3],
        #                                         [4],
        #                                         [5],
        #                                         [1]]))

        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # Model 객체를 이용하여 오차 계산

        print('=====trainModel loss = Model(model, input_tensor, target_tensor, optimizer, criterion) start ')

        loss = Model(Seq2Seq_model, input_tensor, target_tensor, optimizer, criterion)

        print('=====trainModel loss = Model(model, input_tensor, target_tensor, optimizer, criterion) end ')

        print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): for 문 loss = ', loss)
        # loss = 1.7806238174438476

        total_loss_iterations += loss

        if iter % 500 == 0:
            average_loss = total_loss_iterations / 500
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))
        # print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): for 종료 =========')

    torch.save(Seq2Seq_model.state_dict(), '../data/mytraining.pt')

    # print('=====trainModel(model, input_lang, output_lang, pairs, num_iteration=2000): end =========')

    return Seq2Seq_model

################################################################################
# 모델 훈련 (main module)
################################################################################

print('============main module start ==========')

lang_kind_1 = 'eng'  # 입력으로 사용할 영어
lang_kind_2 = 'fra'  # 출력으로 사용할 프랑스어

# 데이터셋 불러오기
input_lang, output_lang, pairs = process_data(lang_kind_1, lang_kind_2)

# print('input_lang {}' .format(input_lang))
# print('output_lang {}' .format(output_lang))
# print('pairs {}' .format(pairs))

# input_lang  = <__main__.Lang object at 0x00000234132C8EF0>
# output_lang = <__main__.Lang object at 0x000002341359DE50>
# pairs = [['i made two.', "j'en ai confectionne deux."]]

randomize = pairs
# print('random sentence {}'.format(randomize))
# random sentence [['i made two.', "j'en ai confectionne deux."]]

input_size = input_lang.n_words
output_size = output_lang.n_words
# 입력과 출력에 대한 단어 수 출력(공백은 계산에서 제외하고 SOS, EOS 2개 문자는 계산 포함)
# print('Input : {} Output : {}'.format(input_size, output_size))

# Input : 5 Output : 6
# input size는 1)sos 2)eos 3)i 4)made 5)two. 로 해서 5임
# output size는 1)sos 2)eos 3)j'en 4)ai 5)confectionne 6)deux. 로 해서 6임

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 1 # 30000 -> 3000
# num_iteration = 75000

# 인코더에 훈련 데이터셋을 입력과 모든 출력과 은닉 상태를 저장 (선언만 한 상태)
#                    23191       512           256          1
# print('encoder = Encoder_Network(input_size, hidden_size, embed_size, num_layers) start')

encoder = Encoder_Network(input_size, hidden_size, embed_size, num_layers)

# print(encoder)
# Encoder(
#   (embedding): Embedding(23191, 256)
#   (gru): GRU(256, 512)
# )

# print('encoder = Encoder_Network(input_size, hidden_size, embed_size, num_layers) end\n')

# 디코더의 첫번째 입력으로 <SOS>토큰이 제공되고 인코더의 마지막 은닉 상태가 디코더의 첫번째 은닉상태로 제공 (선언만 한 상태)
#                     39387        512          256          1
# print('decoder = Decoder_Network(output_size, hidden_size, embed_size, num_layers) start')

decoder = Decoder_Network(output_size, hidden_size, embed_size, num_layers)

# print(decoder)

# print('decoder = Decoder_Network(output_size, hidden_size, embed_size, num_layers) end\n')

# Decoder(
#   (embedding): Embedding(39387, 256)
#   (gru): GRU(256, 512)
#   (out): Linear(in_features=512, out_features=39387, bias=True)
#   (softmax): LogSoftmax(dim=1)
# )

# Encoder_Network(
#   (embedding): Embedding(5, 256) 5개 단어를 256 차원으로 임베딩
#   (gru): GRU(256, 512)
# )
# Decoder_Network(
#   (embedding): Embedding(6, 256)
#   (gru): GRU(256, 512)
#   (out): Linear(in_features=512, out_features=6, bias=True)
#   (softmax): LogSoftmax(dim=1)

# 인코드-디코더 모델(Seq2seq) 모델 생성

# print('Seq2Seq_model = Seq2Seq(encoder, decoder, device).to(device) start')

Seq2Seq_model = Seq2Seq(encoder, decoder, device).to(device)

# print('Seq2Seq_model = Seq2Seq(encoder, decoder, device).to(device) end\n')
# print('Seq2Seq_model =', Seq2Seq_model)

# Seq2Seq_model = Seq2Seq(
#   (encoder): Encoder_Network(
#     (embedding): Embedding(5, 256)
#     (gru): GRU(256, 512)
#   )
#   (decoder): Decoder_Network(
#     (embedding): Embedding(6, 256)
#     (gru): GRU(256, 512)
#     (out): Linear(in_features=512, out_features=6, bias=True)
#     (softmax): LogSoftmax(dim=1)
#   )
# )

print('###################################################################')
print('######## MAIN START ###############################################')
print('###################################################################')

# trainModel은 class가 아닌 일반 함수로서 바로 실행됨
# model = trainModel(Seq2Seq_model, input_lang, output_lang, pairs, num_iteration) 형태가 아닌
# trainModel(Seq2Seq_model, input_lang, output_lang, pairs, num_iteration) 형태로 호출해도 됨

# model = trainModel(Seq2Seq_model, input_lang, output_lang, pairs, num_iteration)
XXX = trainModel(Seq2Seq_model, input_lang, output_lang, pairs, num_iteration)

# 객체의 이름은 임의로 지정해도 됨
# print('XXX =', XXX)

print('###################################################################')
print('######## MAIN END ###############################################')
print('###################################################################')


# print('model =', model)

# model = Seq2Seq(
#   (encoder): Encoder_Network(
#     (embedding): Embedding(5, 256)
#     (gru): GRU(256, 512)
#   )
#   (decoder): Decoder_Network(
#     (embedding): Embedding(6, 256)
#     (gru): GRU(256, 512)
#     (out): Linear(in_features=512, out_features=6, bias=True)
#     (softmax): LogSoftmax(dim=1)
#   )
# )

# Seq2Seq_model = Seq2Seq(
#   (encoder): Encoder_Network(
#     (embedding): Embedding(5, 256)
#     (gru): GRU(256, 512)
#   )
#   (decoder): Decoder_Network(
#     (embedding): Embedding(6, 256)
#     (gru): GRU(256, 512)
#     (out): Linear(in_features=512, out_features=6, bias=True)
#     (softmax): LogSoftmax(dim=1)
#   )
# )

print('============main module end ==========')

################################################################################
# 모델 평가
################################################################################

# def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
#
#     print('=====evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH): start =========')
#
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentences[0])   # 입력 문자열를 텐서로 변환
#         output_tensor = tensorFromSentence(output_lang, sentences[1]) # 출력 문자열을 텐서로 변환
#         decoded_words = []
#         output = model(input_tensor, output_tensor)
#
#         for ot in range(output.size(0)):
#             topv, topi = output[ot].topk(1)  # 각 출력에서 가장 높은 값을 찾아 인덱스로 변환
#
#             if topi[0].item() == EOS_token:
#                 decoded_words.append('<EOS>') # EOS 토큰를 만나면 평가를 멈춤
#                 break
#             else: # 예측 결과를 출력 문자열에 추가
#                 decoded_words.append(output_lang.index2word[topi[0].item()])
#
#     print('=====evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH): end =========')
#
#     return decoded_words

################################################################################
# 훈련 데이터셋으로부터 임의의 문장을 가져와서 모델 평가  (임의로 10개의 데이터를 가져옴)
################################################################################

# def evaluateRandomly(model, input_lang, output_lang, pairs, n=1):
# # def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
#
#     for i in range(n):
#
#         # print('=******************************************************************************=')
#         # print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) start =========')
#         #
#         pair = random.choice(pairs)  # 임의의 문장을 가져온다.
#         print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) === i  =', i,n)
#         print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) === input {}'.format(pair[0]))
#         print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) === output {}'.format(pair[1]))
#
#         output_words = evaluate(model, input_lang, output_lang, pair) # 모델 평가 결과는 output_words에 저장
#         output_sentence = ' '.join(output_words)
#
#         print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) === predicted {}'.format(output_sentence))
#         # print('=====def evaluateRandomly(model, input_lang, output_lang, pairs) end =========')
        # print('=******************************************************************************=')

################################################################################
# 임의의 문장에 대한 평가 결과
################################################################################
#
# print('evaluateRandomly(model, input_lang, output_lang, pairs) start=========')
# evaluateRandomly(model, input_lang, output_lang, pairs)
# print('evaluateRandomly(model, input_lang, output_lang, pairs) end=========')

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
