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


import nltk
nltk.download()
text = nltk.word_tokenize("It is possible ditinguishing cats and dogs")
print(text)

nltk.download('averaged_perceptron_tagger')

nltk.post_tag(text)