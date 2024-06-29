# import numpy as np
# #
# # examp = np.arange(0, 500,3)
# #
# # examp.resize(3, 5,5)
# #
# # print(examp)
# #
#
#
# for count in range(20):
#     count += 1
#     print('count = ', count)
#     print('count % 10 = ', (count % 10))
#     print('not count % 10 = ', not (count % 10))
#
#     # if not (count % 10):
#     #     print('not count % 10 aaaaa = ', not (count % 10))
#
#
# import matplotlib.pyplot as plt
# import mxnet as mx
# from mxnet.gluon.data.vision import tranforms

# import torch
#
# print(torch.__version__)
#
# print(torchaudio.__version__)
#
# print(torvision.__version__)
################################################################################
#9.1.1 자연어처리 용어 및 프로세스
################################################################################

# import nltk
# nltk.download()
# text = nltk.word_tokenize("It is possible distinguishing cats and dogs")
# print(text)
#
# nltk.download('averaged_perceptron_tagger')
#
# nltk.pos_tag(text)
#
# print(nltk.pos_tag(text))

################################################################################
# 9.1.2 자연어처리를 위한 라이브러리 NLTK
################################################################################

# import nltk

# nltk.download('punkt')
#
# string1 = "my favorite subject is math"
# string2 = "my favorite subject is math, english, economic and computer science"
# print(nltk.word_tokenize(string1))
# print(nltk.word_tokenize(string2))

################################################################################
# KoNLPy
################################################################################

# from konlpy.tag import Okt
# okt = Okt()
#
# print(okt.pos('프로그램 설치를 했습니다. 뭐 하나 쉬운 게 없네'))
#
# from konlpy.tag import Komoran
# komoran = Komoran()
# print(komoran.morphs('딥러닝이 쉽나요?, 어렵나요?'))
#
# print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))

################################################################################
#9.2 전처리
#9.2.1 결측치 확인하기
################################################################################

import pandas as pd
df = pd.read_csv('data\class2.csv')
#
# print(df) # 주어진 테이블 확인
# #   Unnamed: 0      id tissue class class2      x      y      r
# # 0           0  mdb000      C  CIRC      N  535.0  475.0  192.0
# # 1           1  mdb001      A  CIRA      N  433.0  268.0   58.0
# # 2           2  mdb002      A  CIRA      I    NaN    NaN    NaN
# # 3           3  mdb003      C  CIRC      B    NaN    NaN    NaN
# # 4           4  mdb004      F  CIRF      I  488.0  145.0   29.0
# # 5           5  mdb005      F  CIRF      B  544.0  178.0   26.0
#
# # isnull()메서더를 사용하여 결측치가 있는지 확인한 후, sum()메서드를 사용하여 결측치가
# # 몇 개인지 합산하여 보여 줌
# print(df.isnull().sum())
#
# # Unnamed: 0    0
# # id            0
# # tissue        0
# # class         0
# # class2        0
# # x             2
# # y             2
# # r             2
# # dtype: int64
#
# # 결측치 비율 확인
# print(df.isnull().sum()/len(df))
# # Unnamed: 0    0.000000
# # id            0.000000
# # tissue        0.000000
# # class         0.000000
# # class2        0.000000
# # x             0.333333
# # y             0.333333
# # r             0.333333
# # dtype: float64

# 결측치 삭제 처리
# df = df.dropna(how='all') # 모든 행이 NaN일 때만 삭제
# print(df)
# #
# #    Unnamed: 0      id tissue class class2      x      y      r
# # 0           0  mdb000      C  CIRC      N  535.0  475.0  192.0
# # 1           1  mdb001      A  CIRA      N  433.0  268.0   58.0
# # 2           2  mdb002      A  CIRA      I    NaN    NaN    NaN
# # 3           3  mdb003      C  CIRC      B    NaN    NaN    NaN
# # 4           4  mdb004      F  CIRF      I  488.0  145.0   29.0
# # 5           5  mdb005      F  CIRF      B  544.0  178.0   26.0

# 데이터에 하나라도 NaN 값이 있으면 행을 삭제
# df = df.dropna()
# print(df)
#
# #    Unnamed: 0      id tissue class class2      x      y      r
# # 0           0  mdb000      C  CIRC      N  535.0  475.0  192.0
# # 1           1  mdb001      A  CIRA      N  433.0  268.0   58.0
# # 4           4  mdb004      F  CIRF      I  488.0  145.0   29.0
# # 5           5  mdb005      F  CIRF      B  544.0  178.0   26.0

# 결측치를 0으로 채우기
# df2 = df.fillna(0)
# print(df2)
# #
# #    Unnamed: 0      id tissue class class2      x      y      r
# # 0           0  mdb000      C  CIRC      N  535.0  475.0  192.0
# # 1           1  mdb001      A  CIRA      N  433.0  268.0   58.0
# # 2           2  mdb002      A  CIRA      I    0.0    0.0    0.0
# # 3           3  mdb003      C  CIRC      B    0.0    0.0    0.0
# # 4           4  mdb004      F  CIRF      I  488.0  145.0   29.0
# # 5           5  mdb005      F  CIRF      B  544.0  178.0   26.0

# 결측치를 평균으로 채우기
# df['x'].fillna(df['x'].mean(), inplace=True) # x열에 대해 평균값(500.0)으로 NaN값이 채워짐
# print(df)
#
# #    Unnamed: 0      id tissue class class2      x      y      r
# # 0           0  mdb000      C  CIRC      N  535.0  475.0  192.0
# # 1           1  mdb001      A  CIRA      N  433.0  268.0   58.0
# # 2           2  mdb002      A  CIRA      I  500.0    NaN    NaN
# # 3           3  mdb003      C  CIRC      B  500.0    NaN    NaN
# # 4           4  mdb004      F  CIRF      I  488.0  145.0   29.0
# # 5           5  mdb005      F  CIRF      B  544.0  178.0   26.0

#9.2.2 토큰화

# 문장 토큰화
# from nltk import sent_tokenize
# text_sample = 'Natural Language Processing, or NLP, is the process of extracting ' \
#               'the meaning, or intent, behind human language. In the field of ' \
#               'Conversational artificial intelligence (AI), NLP allows machines ' \
#               'and applications to understand the intent of human language inputs, ' \
#               'and then generate appropriate responses, resulting in a natural conversation flow.'
# tokenized_sentences = sent_tokenize(text_sample)
# print(tokenized_sentences)
# #
# # ['Natural Language Processing, or NLP, is the process of extracting the meaning, or intent, behind human language.',
# #  'In the field of Conversational artificial intelligence (AI), NLP allows machines and applications to understand the '
# #  'intent of human language inputs, and then generate appropriate responses, resulting in a natural conversation flow.']

# 단어 토큰화
# from nltk import word_tokenize
# sentence = " This book is for deep learning learners"
# words = word_tokenize(sentence)
# print(words)
#
# # ['This', 'book', 'is', 'for', 'deep', 'learning', 'learners']

# 아포스트로피가 포함한 문장에서 단어 토큰화
# from nltk.tokenize import WordPunctTokenizer
# sentence = "it’s nothing that you don’t already know except most people aren’t aware of how their inner world works."
# words = WordPunctTokenizer().tokenize(sentence)
# print(words)
#
# # ['it', '’', 's', 'nothing', 'that', 'you', 'don', '’', 't', 'already', 'know', 'except', 'most', 'people', 'aren', '’', 't', 'aware', 'of', 'how', 'their', 'inner', 'world', 'works', '.']

################################################################################
#한국어 토큰화 예제
################################################################################

# #라이브러리 호출 및 데이터셋 준비
# import csv
# from konlpy.tag import Okt
# from gensim.models import word2vec
#
# f = open(r'data\ratings_train.txt', 'r', encoding='utf-8')
# rdr = csv.reader(f, delimiter='\t')
# rdw = list(rdr)
# f.close()
#
#
# # 오픈 소스 한글 형태소 분석기 호출
# twitter = Okt()  # 오픈 소스 한글 형태소 분석기 Twitter(Okt))를 사용
#
# result = []
# for line in rdw:   # 테스트를 한 줄씩 처리
#     malist = twitter.pos( line[1], norm=True, stem=True)  # 형태소 분석
#     r = []
#     for word in malist:
#         if not word[1] in ["Josa","Eomi","Punctuation"]: # 조사, 어미, 문장부호는 제외하고 처리
#             r.append(word[0])
#     rl = (" ".join(r)).strip()  # 형태소 사
#     result.append(rl)
#     # print(rl)
#
# # 형태소 저장
# with open("NaverMovie.nlp",'w', encoding='utf-8') as fp:
#     fp.write("\n".join(result))
#
# # Word2Vec 모델 생성
# mData = word2vec.LineSentence("NaverMovie.nlp")
# mModel =word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
# mModel.save("NaverMovie.model")

################################################################################
#9.2.3 불용어 제거
################################################################################

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
#
# sample_text = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
# text_tokens = word_tokenize(sample_text)
#
# tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
# print("불용어 제거 미적용:", text_tokens, '\n')
# print("불용어 제거 적용:",tokens_without_sw)
#
# # 불용어 제거 미적용: ['One', 'of', 'the', 'first', 'things', 'that', 'we', 'ask', 'ourselves', 'is', 'what', 'are', 'the', 'pros', 'and', 'cons', 'of', 'any', 'task', 'we', 'perform', '.']
# #
# # 불용어 제거 적용: ['One', 'first', 'things', 'ask', 'pros', 'cons', 'task', 'perform', '.']


#9.2.4 어간 추출

# #포터 알고리즘
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
#
# print(stemmer.stem('obesses'),stemmer.stem('obssesed'))
# print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
# print(stemmer.stem('national'), stemmer.stem('nation'))
# print(stemmer.stem('absentness'), stemmer.stem('absently'))
# print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))
#
# # obess obsses
# # standard standard
# # nation nation
# # absent absent
# # tribal tribalic

#랭커스터 알고리즘
# from nltk.stem import LancasterStemmer
# stemmer = LancasterStemmer()
#
# print(stemmer.stem('obsesses'),stemmer.stem('obsessed'))
# print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
# print(stemmer.stem('national'), stemmer.stem('nation'))
# print(stemmer.stem('absentness'), stemmer.stem('absently'))
# print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))
#
# # obsess obsess
# # standard standard
# # nat nat
# # abs abs
# # trib trib

#표제어 추출(Lemmatization)
# import nltk
# nltk.download('wordnet')
# from nltk.stem import LancasterStemmer
# stemmer = LancasterStemmer()
#
# from nltk.stem import WordNetLemmatizer  # 표제어 추출 라이브러리
# lemma = WordNetLemmatizer()
#
# print(stemmer.stem('obsesses'),stemmer.stem('obsessed'))
# print(lemma.lemmatize('standardizes'),lemma.lemmatize('standardization'))
# print(lemma.lemmatize('national'), lemma.lemmatize('nation'))
# print(lemma.lemmatize('absentness'), lemma.lemmatize('absently'))
# print(lemma.lemmatize('tribalical'), lemma.lemmatize('tribalicalized'))
#
# # obsess obsess
# # standardizes standardization
# # national nation
# # absentness absently
# # tribalical tribalicalized

# 품사 정보가 추가된 표제어 추출
# from nltk.stem import WordNetLemmatizer  # 표제어 추출 라이브러리
# lemma = WordNetLemmatizer()
#
# print(lemma.lemmatize('obsesses', 'v'),lemma.lemmatize('obsessed','a'))
# print(lemma.lemmatize('standardizes','v'),lemma.lemmatize('standardization','n'))
# print(lemma.lemmatize('national','a'), lemma.lemmatize('nation','n'))
# print(lemma.lemmatize('absentness','n'), lemma.lemmatize('absently','r'))
# print(lemma.lemmatize('tribalical','a'), lemma.lemmatize('tribalicalized','v'))
#
# # obsess obsessed
# # standardize standardization
# # national nation
# # absentness absently
# # tribalical tribalicalized

