# # 텍스트 토큰화
# import spacy
# nlp = spacy.load('en_core_web_sm')
# text = "Mary, don't slap the green witch."
# print([str(token) for token in nlp(text.lower())])
#
# from nltk.tokenize import TweetTokenizer
# tweet=u"Snow White and the Seven Degrees#MakeAMoieCold@midnight:-)"
# tokenizer = TweetTokenizer()
# print(tokenizer.tokenize(tweet.lower()))
#
# # 텍스트에서 n-그램 만들기
# def n_grams(text,n):
#     '''
#     takes tokens or text, returns a list of n-grams
#     '''
#     print("len(text) = ", len(text))
#     # len(text) = 7
#     # range(len(text)-n+1) = 7-3+1 =  5
#     return [text[i:i+n] for i in range(len(text)-n+1)]
#
# cleaned = ['mary', ',', "n't", 'slap', 'green', 'witch', '.']
#
# print(n_grams(cleaned,3))
# # [['mary', ',', "n't"], [',', "n't", 'slap'], ["n't", 'slap', 'green'], ['slap', 'green', 'witch'], ['green', 'witch', '.']]
#
# # 표제어 추출 : 단어를 표제어로 바꿉니다.
# import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"he was running late")
# for token in doc:
#     print('{} --> {}'.format(token,token.lemma_))
#
# # he --> he
# # was --> be
# # running --> run
# # late --> late
#
# # 품사 태깅
# import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"Mary slapped the green witch.")
# for token in doc:
#     print('{} - {}'.format(token, token.pos_))
#
# # Mary - PROPN
# # slapped - VERB
# # the - DET
# # green - ADJ
# # witch - NOUN
# # . - PUNCT
#
# # 명사구(NLP) 부분 구문 분석
# import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"Mary slapped the green witch.")
# for chunk in doc.noun_chunks:
#     print('{} - {} '.format(chunk, chunk.label_))
#
# # Mary - NP
# # the green witch - NP
#
# #시그모이드 활성화 함수
# import torch
# import matplotlib.pyplot as plt
#
# x = torch.range(-5., 5., 0.1)
# y = torch.sigmoid(x)
# plt.plot(x.numpy(), y.numpy())
# plt.show()
#
# # 하이퍼볼릭 탄젠트 활성화 함수
# import torch
# import matplotlib.pyplot as plt
#
# x = torch.range(-5., 5., 0.1)
# y = torch.tanh(x)
# plt.plot(x.numpy(), y.numpy())
# plt.show()
#
# # 렐루 활성화함수
# import torch
# import matplotlib.pyplot as plt
#
# x = torch.range(-5., 5., 0.1)
# y = torch.relu(x)
# plt.plot(x.numpy(), y.numpy())
# plt.show()
#
# # PReLU 활성화함수
# import torch
# import matplotlib.pyplot as plt
#
# prelu = torch.nn.PReLU(num_parameters=1)
# x = torch.range(-5., 5., 0.1)
# y = prelu(x)
# plt.plot(x.numpy(), y.detach().numpy())
# plt.show()
#
# # 소프트맥스 활성화 함수
# import torch.nn as nn
# import torch
#
# softmax = nn.Softmax(dim=1)
# x_input = torch.randn(1,3)
# y_output = softmax(x_input)
#
# print(x_input)
# print(y_output)
# print(torch.sum(y_output,dim=1))

# # 평균 제곱 오차 손실
# import torch
# import torch.nn as nn
#
# mse_loss = nn.MSELoss()
# outputs = torch.randn(3,5, requires_grad=True)
# targets = torch.randn(3,5)
# loss = mse_loss(outputs, targets)
# print(loss)
#
# # tensor(1.6269, grad_fn=<MseLossBackward0>)

# # 크로스 엔트로피 손실
# import torch
# import torch.nn as nn
#
# ce_loss = nn.CrossEntropyLoss()
# outputs = torch.randn(3,5, requires_grad=True)
# targets = torch.tensor([1, 0, 3], dtype=torch.int64)
# loss = ce_loss(outputs, targets)
# print(loss)
#
# # tensor(1.9082, grad_fn=<NllLossBackward0>)

# 이진 크로스 엔트로피 손실
# import torch
# import torch.nn as nn
# bce_loss = nn.BCELoss()
# sigmoid = nn.Sigmoid()
# probabilities = sigmoid(torch.randn(4,1, requires_grad=True))
# targets = torch.tensor([1,0,1,0], dtype=torch.float32).view(4,1)
# loss = bce_loss(probabilities, targets)
# print(probabilities)
# print(loss)
#
# # tensor([[0.2665],
# #         [0.3571],
# #         [0.6895],
# #         [0.8497]], grad_fn=<SigmoidBackward0>)
# # tensor(1.0079, grad_fn=<BinaryCrossEntropyBackward0>)
# #
