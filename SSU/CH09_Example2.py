# Import Library
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

## DATA SET
dataset = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)

input_data  = dataset[:,1:]
target_data  = dataset[:,0]
print("input_data.shape = ", input_data.shape)
print("target_data.shape = ", target_data.shape)

label = np.zeros((60000,2))
for i in range(60000):
    if target_data[i]== 1.:
        label[i,1]=1
    if target_data[i]== 0.:
        label[i,0]=1

## Activate Fuction Sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))

## Basic Deep Learning Network
class BasicDeepLearning:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # hidden layer weight init
        self.W2 = np.random.random((self.input_nodes, self.hidden_nodes))
        # output layer weight init
        self.W3 = np.random.random((self.hidden_nodes, self.output_nodes))

        # output layer
        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])

        # hidden layer
        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])

        # input layer
        self.Z1 = np.zeros([1, input_nodes])
        self.A1 = np.zeros([1, input_nodes])

        # learning rate
        self.learning_rate = learning_rate
        print("self.Z1.shape = ",self.Z1.shape)
        print("self.A1.shape = ",self.A1.shape)
        print("self.Z2.shape = ",self.Z2.shape)
        print("self.A2.shape = ",self.A2.shape)
        print("self.Z3.shape = ",self.Z3.shape)
        print("self.A3.shape = ",self.A3.shape)
        print("self.W2.shape = ",self.W2.shape)
        print("self.W3.shape = ",self.W3.shape)
        '''
self.A1.shape =  (1, 784)
self.Z2.shape =  (1, 100)
self.A2.shape =  (1, 100)
self.Z3.shape =  (1, 10)
self.A3.shape =  (1, 10)
self.W2.shape =  (784, 100)
self.W3.shape =  (100, 10)
        '''
    ## Feed Forward
    def feed_forward(self):
        # input layer
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # hidden layer
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = sigmoid(self.Z2)

        # output layser
        self.Z3 = np.dot(self.A2, self.W3)
        self.A3 = sigmoid(self.Z3)

        return self.A3

    ## loss
    def loss_val(self):
        loss = ((self.A3 - self.target_data) ** 2)
        loss = np.mean(loss)

        return loss

    ## Train(Backwoard, Weight Update)
    def train(self, input_data, target_data):
        self.target_data = target_data
        self.input_data = input_data

        # feedforward
        self.feed_forward()

        # stage1 (output → hidden weight update)
        loss_3 = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)

        # stage2 (hidden → input weight update)
        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1 - self.A2)
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)

## Parameter Settinga
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.01
epochs=1

## Training for optimization
loss = []
bdl = BasicDeepLearning(input_nodes, hidden_nodes, output_nodes, learning_rate)

for i in range(epochs):
    bdl.train( input_data, target_data)
    loss.append(bdl.loss_val())
    print(bdl.loss_val())
