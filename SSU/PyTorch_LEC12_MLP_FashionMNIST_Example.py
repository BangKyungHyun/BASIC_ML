import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

train_dataset = datasets.FashionMNIST(root='FashionMNIST_data/', train=True,
                                      # 학습 데이터
                                      transform=transforms.ToTensor(),
                                      # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                                      download=True)

test_dataset = datasets.FashionMNIST(root='FashionMNIST_data/', train=False,
                                     # 테스트 데이터
                                     transform=transforms.ToTensor(),
                                     # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                                     download=True)

print(len(train_dataset))

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

print(len(train_dataset), len(validation_dataset), len(test_dataset))

BATCH_SIZE = 32

train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,shuffle=True)

validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, data):
        data = self.flatten(data)
        data = self.fc1(data)
        data = self.relu(data)
        data = self.dropout(data)
        logits = self.fc2(data)
        return logits

model = MyDeepLearningModel()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    total_train_batch = len(dataloader)

    print('len(dataloader) = ',len(dataloader))
    # 1594  = 51000/32

    for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        x_train = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
        y_train = labels

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)  # label 열 사이즈 같음

        # print('y_train.size(0) =', y_train.size(0))
        # y_train.size(0) = 32

        train_correct += ((torch.argmax(outputs, 1)==y_train)).sum().item() # 예측한 값과 일치한 값의 합

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100*train_correct / train_total

    print('train_avg_loss, train_loss_sum,  total_train_batch =', train_avg_loss,  train_loss_sum, total_train_batch)
    print('train_avg_accuracy, train_correct, train_total =', train_avg_accuracy, train_correct, train_total)
    # len(dataloader) =  1594
    # train_avg_loss, train_loss_sum,  total_train_batch = 0.9933313936406426 1583.3702414631844 1594
    # train_avg_accuracy, train_correct, train_total = 67.69019607843137 34522 51000
    # epoch: 01 train loss = 0.9933 train accuracy = 67.6902 validation loss = 0.6526 validation accuracy = 78.0556

    return (train_avg_loss, train_avg_accuracy)


def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad(): #미분하지 않겠다는 것

        val_loss_sum = 0
        val_correct=0
        val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_val = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)  # label 열 사이즈 같음
            val_correct += ((torch.argmax(outputs, 1)==y_val)).sum().item() # 예측한 값과 일치한 값의 합

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100*val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)

def model_test(dataloader, model):

    model.eval()

    with torch.no_grad(): #test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.

        test_loss_sum = 0
        test_correct=0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_test = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_test = labels

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()

            test_total += y_test.size(0)  # label 열 사이즈 같음
            test_correct += ((torch.argmax(outputs, 1)==y_test)).sum().item() # 예측한 값과 일치한 값의 합

        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100*test_correct / test_total

        print('accuracy:', test_avg_accuracy)
        print('loss:', test_avg_loss)


from datetime import datetime

train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

start_time = datetime.now()

EPOCHS = 20

for epoch in range(EPOCHS):
    # ==============  model train  ================
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader,
                                                     model, loss_function,
                                                     optimizer)  # training

    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)
    # =============================================

    # ============  model evaluation  ==============
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader,
                                                    model, loss_function,
                                                    optimizer)  # evaluation

    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)
    # ============  model evaluation  ==============

    print('epoch:', '%02d' % (epoch + 1),
          'train loss =', '{:.4f}'.format(train_avg_loss), 'train accuracy =',
          '{:.4f}'.format(train_avg_accuracy),
          'validation loss =', '{:.4f}'.format(val_avg_loss),
          'validation accuracy =', '{:.4f}'.format(val_avg_accuracy))

end_time = datetime.now()

print('elapsed time => ', end_time - start_time)

model_test(test_dataset_loader, model)

import matplotlib.pyplot as plt

plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(val_accuracy_list, label='validation accuracy')

plt.legend()

plt.show()


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss_list, label='train')
plt.plot(val_loss_list, label='validation')
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(train_accuracy_list, label='train')
plt.plot(val_accuracy_list, label='validation')
plt.legend()

plt.show()

