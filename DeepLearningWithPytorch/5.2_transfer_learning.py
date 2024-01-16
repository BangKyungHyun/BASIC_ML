#5.3.1 특성 추출 기법
################################################################################
# 라이브러리 호출
################################################################################
import os
import time
import copy
import glob
import cv2
import shutil

import torch
import torchvision   # 컴퓨터 비젼 용도의 패키지
import torchvision.transforms as transforms # 데이터 전처리를 위해 사용하는 패키지
import torchvision.models as models # 다양한 파이토지 네트워크를 사용할 수 있도록 도와주는 패키지
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

################################################################################
# 이미지 데이터 전처리 방법 정의
################################################################################
data_path = 'data/catanddog/train'

# torchvision.transform은 이미지 데이터를 변환하여 모델(네트워크) 입력으로 사용할 수 있게
# 변환해 줌
transform = transforms.Compose(
    [
        # 이미지 크기를 조정, 즉 256* 256 크기로 이미지 데이터를 조정
        transforms.Resize([256, 256]),
        # 이미지를 랜덤한 크기 및 비율로 자름
        transforms.RandomResizedCrop(224),
        # 이미지를 랜덤하게 수편을 뒤집음
        transforms.RandomHorizontalFlip(),
        # 이미지 데이터를 텐서로 변환
        transforms.ToTensor(),
    ])
# datasets.ImageFolder는 데이터로더가 데이터를 불러올 대상(혹은 경로)과 방법(transform)
# (혹은 전처리) 를 정의
train_dataset = torchvision.datasets.ImageFolder(
    data_path,
    # 이미지 데이터에 대한 전처리
    transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, # 데이터셋을 지정
    batch_size=32, # 한번에 불러올 데이터양을 결정하는 배치크기를 정의
    num_workers=8, # 데이터를 불러올 때 하위 프로세스를 몇개 사용하는 설정
    shuffle=True
)

print('len(train_dataset) = ', len(train_dataset))

# 반복자(iterator, for 구문과 같은 효과)를 사용하려면 iter()와 next()가 필요
# iter()는 전달된 데이터의 반복자를 꺼내 반환하며, next()는 그 반복자가 다음에 출력해야 할
# 요소를 반환. 즉 iter()로 반복자를 구하고 그 반복자를 next()에 전달하여 차례대로 꺼낼 수 있음
# 코드에서 반복자는 train_loader가 되기 때문에 train_loader에서 samples와 labels의 값을
# 순차적으로 꺼내 저장함

samples, labels = iter(train_loader).next()

# 개와 고양이에 대한 클래스로 구성
classes = {0:'cat', 1:'dog'}
fig = plt.figure(figsize=(16,24))

# 24개 이미지 데이터 출력
for i in range(24):
    a = fig.add_subplot(4,6,i+1)
    # 레이블 정보(클래스)를 함께 출력
    a.set_title(classes[labels[i].item()])
    a.axis('off')
    # np.transpose는 행과 열을 바꿈으로써 행렬의 차이를 바꾸어 줌
    a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))

plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

resnet18 = models.resnet18(pretrained=True)

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

set_parameter_requires_grad(resnet18)

resnet18.fc = nn.Linear(512, 2)

for name, param in resnet18.named_parameters():
    if param.requires_grad:
        print(name, param.data)

model = models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, 2)
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters())
cost = torch.nn.CrossEntropyLoss()
print(model)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=13, is_train=True):
    since = time.time()
    acc_history = []
    loss_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        torch.save(model.state_dict(), os.path.join('data/catanddog/',
                                                    '{0:0=2d}.pth'.format(
                                                        epoch)))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    return acc_history, loss_history

params_to_update = []
for name, param in resnet18.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)

optimizer = optim.Adam(params_to_update)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
train_acc_hist, train_loss_hist = train_model(resnet18, train_loader, criterion, optimizer, device)

test_path = 'data/catanddog/test'

transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])
test_dataset = torchvision.datasets.ImageFolder(
    root=test_path,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=1,
    shuffle=True
)

print(len(test_dataset))

def eval_model(model, dataloaders, device):
    since = time.time()
    acc_history = []
    best_acc = 0.0

    saved_models = glob.glob('data/catanddog/' + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)

    for model_path in saved_models:
        print('Loading model', model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        running_corrects = 0

        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            running_corrects += preds.eq(labels.cpu()).int().sum()

        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print('Acc: {:.4f}'.format(epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        print()

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                          time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))

    return acc_history

val_acc_hist = eval_model(resnet18, test_loader, device)

plt.plot(train_acc_hist)
plt.plot(val_acc_hist)
plt.show()

plt.plot(train_loss_hist)
plt.show()

def im_convert(tensor):
    image=tensor.clone().detach().numpy()
    image=image.transpose(1,2,0)
    image=image*(np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5)))
    image=image.clip(0,1)
    return image

classes = {0:'cat', 1:'dog'}

dataiter=iter(test_loader)
images,labels=dataiter.next()
output=model(images)
_,preds=torch.max(output,1)

fig=plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax=fig.add_subplot(2,10,idx+1,xticks=[],yticks=[])
    plt.imshow(im_convert(images[idx]))
    a.set_title(classes[labels[i].item()])
    ax.set_title("{}({})".format(str(classes[preds[idx].item()]),str(classes[labels[idx].item()])),color=("green" if preds[idx]==labels[idx] else "red"))
plt.show()
plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)