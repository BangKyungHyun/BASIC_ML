import numpy as np
import sys
import os
from array import array

from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#파일 읽기
fp_image = open('train-images-idx3-ubyte','rb')
fp_label = open('train-labels-idx1-ubyte','rb')

#사용할 변수 초기화
img = np.zeros((28,28)) #이미지가 저장될 부분
lbl = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
d = 0
l = 0
index=0

s = fp_image.read(16)    #read first 16byte
l = fp_label.read(8)     #read first  8byte

#숫자 데이터를 읽어서 해당하는 데이터를 지정하고 출력

k=1 #테스트용 index
#read mnist and show number
while True:
    s = fp_image.read(784) #784바이트씩 읽음
    l = fp_label.read(1) #1바이트씩 읽음

    if not s:
        break;
    if not l:
        break;

    index = int(l[0])
    #print(k,":",index)

#unpack
    img = np.reshape( unpack(len(s)*'B',s), (28,28))
    lbl[index].append(img) #각 숫자영역별로 해당이미지를 추가
    k=k+1

#print(img)

plt.imshow(img,cmap = cm.binary) #binary형태의 이미지 설정
plt.show()

#print(np.shape(lbl)) #label별로 잘 지정됬는지 확인

print("read done")