# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 2017
Neural network example
"""
import numpy as np

import mnist_loader
from helpers import png2input
from network import Network

np.random.seed(420)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

nn = Network([784, 30, 10])
nn.SGD(training_data, 30, 10, 2.5, test_data=test_data)
nn.save('data/example.npz')
# nn.load('data/example.npz')

print('score = {:.2f} %'.format(nn.evaluate(test_data)/100))


pix_data = png2input('data/example.png')

results = nn.feedforward(pix_data)
print('You wrote: {}'.format(np.argmax(results)))
for i in range(10):
    print('{}: {:4.1f} %'.format(i, 100*results[i][0]))