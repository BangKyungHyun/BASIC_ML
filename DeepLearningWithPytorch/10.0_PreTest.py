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

# #10.1.1 희소표현(Sparse Representation)
#
# one hot encoding 사용
import pandas as pd
class2=pd.read_csv("data/class2.csv")
# class2=pd.read_csv("../chap10/data/class2.csv")

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2['class2'])
print(train_x)

#10.1.2 횟수기반 임베딩 Counter Vector

# 코퍼스에 카운터 벡터 사용
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is last chance.',
    'and if you do not have this chance.',
    'you will never get any chance.',
    'will you do get this one?',
    'please, get this chance',
]
vect = CountVectorizer()
vect.fit(corpus)

print(vect.vocabulary_)
#
# {'this': 13, 'is': 7, 'last': 8, 'chance': 2, 'and': 0, 'if': 6, 'you': 15, 'do': 3, 'not': 10, 'have': 5, 'will': 14, 'never': 9, 'get': 4, 'any': 1, 'one': 11, 'please': 12}

# 배열 변환
print(vect.transform(['you will never get any chance.']).toarray())
print(vect.transform(['you will never get any chance one.']).toarray())

# [[0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1]]

# 불용어를 제거한 카운터 벡터
# vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
# print(vect.vocabulary_)

# {'last': 6, 'chance': 1, 'if': 5, 'you': 11, 'do': 2, 'not': 8, 'have': 4, 'will': 10, 'never': 7, 'get': 3, 'any': 0, 'one': 9}

#TF-IDF

# from sklearn.feature_extraction.text import TfidfVectorizer
# doc = ['I like machine learning', 'I love deep learning', 'I run everyday']
# tfidf_vectorizer = TfidfVectorizer(min_df=1)
# tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
# doc_distance = (tfidf_matrix * tfidf_matrix.T)
# print ('유사도를 위한', str(doc_distance.get_shape()[0]), 'x', str(doc_distance.get_shape()[1]), 'matrix를 만들었습니다.')
# print(doc_distance.toarray())
#
# 유사도를 위한 3 x 3 matrix를 만들었습니다.
# [[1.       0.224325 0.      ]
#  [0.224325 1.       0.      ]
#  [0.       0.       1.      ]]

#10.1.3 예측기반 임베딩 Word2Vec

# 데이터셋을 메모리로 로딩하고 토근화 적용
# from nltk.tokenize import sent_tokenize, word_tokenize
# import warnings
#
# warnings.filterwarnings(action='ignore')
# import gensim
# from gensim.models import Word2Vec
#
# sample = open("data/peter.txt", "r", encoding='UTF8')
# s = sample.read()
#
# f = s.replace("\n", " ")  # 줄바꿈을 공백으로 변환
# data = []
#
# for i in sent_tokenize(f):    #로딩한 파일의 각 문장마다 반복
#     temp = []
#     for j in word_tokenize(i):  # 문장을 단어로 토근화
#         temp.append(j.lower())  # 토근화된 단어를 소문자로 변환하여 temp에 저장
#     data.append(temp)

# print(data)

# [['once', 'upon', 'a', 'time', 'in', 'london', ',', 'the', 'darlings', 'went', 'out', 'to', 'a', 'dinner', 'party', 'leaving',
#   'their', 'three', 'children', 'wendy', ',', 'jhon', ',', 'and', 'michael', 'at', 'home', '.'],
#  ['after', 'wendy', 'had', 'tucked', 'her', 'younger', 'brothers', 'jhon', 'and', 'michael', 'to', 'bed', ',', 'she', 'went', 'to', 'read', 'a', 'book', '.'],

#CBOW

# # 데이터셋에 CBOW 적용후 peter 와 wedny 유사성 확인
# model1 = gensim.models.Word2Vec(data, min_count = 1,
#                               vector_size = 100, window = 5, sg=0)
# print("Cosine similarity between 'peter' " +
#                  "wendy' - CBOW : ",
#       model1.wv.similarity('peter', 'wendy'))
#
# # Cosine similarity between 'peter' wendy' - CBOW :  0.07439385
#
# # 데이터셋에 CBOW 적용후 peter 와 hook 유사성 확인
#
# print("Cosine similarity between 'peter' " +
#                  "hook' - CBOW : ",
#       model1.wv.similarity('peter', 'hook'))
#
# # Cosine similarity between 'peter' hook' - CBOW :  0.027709909
#
# #Skip-gram
# # 데이터셋에 skip-gram 적용후 peter 와 wedny 유사성 확인
#
# model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
#                                               window = 5, sg = 1)
# # print("Cosine similarity between 'peter' " +
#           "wendy' - Skip Gram : ",
#     model2.wv.similarity('peter', 'wendy'))
#
# # Cosine similarity between 'peter' wendy' - Skip Gram :  0.4008868

# print("Cosine similarity between 'peter' " +
#             "hook' - Skip Gram : ",
#       model2.wv.similarity('peter', 'hook'))
#
# # Cosine similarity between 'peter' hook' - Skip Gram :  0.52016735

# #FastText
#
# from gensim.test.utils import common_texts
# from gensim.models import FastText
#
# model = FastText('data/peter.txt', vector_size=4, window=3, min_count=1, epochs=10)
#
# # peter 와 wendy에 대한 코사인 유사도
#
# sim_score = model.wv.similarity('peter', 'wendy')
# print(sim_score)
# # 0.45924556
# # peter 와 hook에 대한 코사인 유사도
#
# sim_score = model.wv.similarity('peter', 'hook')
# print(sim_score)
# # 0.04382518

# from gensim.models import KeyedVectors
#
# model_kr = KeyedVectors.load_word2vec_format('data/wiki.ko.vec')
#
# find_similar_to = '노력'
#
# for similar_word in model_kr.similar_by_word(find_similar_to):
#     print("Word: {0}, Similarity: {1:.2f}".format(
#         similar_word[0], similar_word[1]
#     ))
#
# # Word: 노력함, Similarity: 0.80
# # Word: 노력중, Similarity: 0.75
# # Word: 노력만, Similarity: 0.72
# # Word: 노력과, Similarity: 0.71
# # Word: 노력의, Similarity: 0.69
# # Word: 노력가, Similarity: 0.69
# # Word: 노력이나, Similarity: 0.69
# # Word: 노력없이, Similarity: 0.68
# # Word: 노력맨, Similarity: 0.68
# # Word: 노력보다는, Similarity: 0.68

# similarities = model_kr.most_similar(positive=['동물', '육식동물'], negative=['사람'])
# print(similarities)
#
# # [('초식동물', 0.7804122567176819), ('거대동물', 0.7547270059585571), ('육식동물의', 0.7547166347503662), ('유두동물', 0.7535113096237183), ('반추동물', 0.7470757365226746), ('독동물', 0.7466291785240173), ('육상동물', 0.746031641960144), ('유즐동물', 0.7450904250144958), ('극피동물', 0.7449344396591187), ('복모동물', 0.7424346208572388)]

#10.1.4 횟수/예측기반 임베딩
#GloVe

# import os
#
# import numpy as np
# # %matplotlib notebook
# import matplotlib.pyplot as plt
# # plt.style.use('ggplot')
# from sklearn.decomposition import PCA
# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
#
# glove_file = datapath('C:\\Users\\bangkh21\\PycharmProjects\\BASIC_ML\\DeepLearningWithPytorch\data\\glove.6B.100d.txt')
# word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
# print(glove2word2vec(glove_file, word2vec_glove_file))
# # (400000, 100)
#
# # bill과 유사한 단어의 리스트를 반환
# # load_word2vec_format() 메서드를 이용하여 word2vec.c 형식을 벡터를 가져옴
# model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
#
# # 단어(bill) 기준으로 가장 유사한 단어들의 리스트를 보여줌
# print(model.most_similar('bill'))
# # [('legislation', 0.8072139620780945), ('proposal', 0.730686366558075), ('senate', 0.7142541408538818), ('bills', 0.704440176486969), ('measure', 0.6958035230636597), ('passed', 0.6906245350837708), ('amendment', 0.6846878528594971), ('provision', 0.6845566630363464), ('plan', 0.6816462874412537), ('clinton', 0.6663140058517456)]
#
# # 단어(cherry) 기준으로 가장 유사한 단어들의 리스트를 보여줌
# print(model.most_similar('cherry'))
# # [('peach', 0.688809871673584), ('mango', 0.6838189959526062), ('plum', 0.6684104204177856), ('berry', 0.6590359807014465), ('grove', 0.6581552028656006), ('blossom', 0.6503506302833557), ('raspberry', 0.6477391719818115), ('strawberry', 0.6442098021507263), ('pine', 0.6390928626060486), ('almond', 0.6379212737083435)]
# [
# # 단어(cherry) 와 관련성이 없는 단어의 리스트를 반환
# print(model.most_similar(negative='cherry'))
# # [('kazushige', 0.48343509435653687), ('askerov', 0.47781863808631897), ('lakpa', 0.46915262937545776), ('ex-gay', 0.45713329315185547), ('tadayoshi', 0.4522107243537903), ('turani', 0.44810065627098083), ('saglam', 0.4469599425792694), ('aijun', 0.4435270130634308), ('adjustors', 0.44235295057296753), ('nyum', 0.4423118531703949)]
#
# # woman, king과 유사성이 높으면서 man과 관련성이 없는 단어를 반환
# result = model.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))
# # queen: 0.7699
#
# def analogy(x1, x2, y1):
#     result = model.most_similar(positive=[y1, x2], negative=[x1])
#     return result[0][0]
#
# # 'australia', 'beer', 'france'와 관련성이 있는 단어를 유추
# print(analogy('australia', 'beer', 'france'))
# # champagne
#
# # 'tall', 'tallest', 'long' 단어를 기반으로 새로운 단어를 유추
# print(analogy('tall', 'tallest', 'long'))
# # longest
#
# # breakfast cereal dinner lunch 중 유사도가 낮은 단어를 반환
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))
# # cereal
#
