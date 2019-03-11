import numpy as np
from PIL import Image


def normalize(some_list):
    max_value = max(some_list)
    if not max_value == 0:
        some_list = [i/max_value for i in some_list]
    return some_list


def input2png(input_data, filename):
    mode = 'L'
    im = Image.new(mode, (28, 28))
    pix = im.load()
    for y in range(28):
        for x in range(28):
            value = int(255*(1-input_data[28*y+x]))
            pix[x,y] = (value,)
    im.save(filename)


def png2input(filename):
    im = Image.open(filename)
    pix = im.load()
    pix_data = np.array([[(1-(pix[x, y]/255))] for y in range(28) for x in range(28)])
    return normalize(pix_data)