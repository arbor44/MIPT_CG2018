import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import math

def gamma_correction(im, gamma):
    if gamma == 1:
        return im
    m = im.max()
    return m * (im / m) ** gamma


def convert_to_grayscale(im, weights=[0.3, 0.6, 0.1]):
    gray = np.zeros(im.shape[:2])
    for i in range(3):
        gray = gray + im[:, :, i]*weights[i]
    
    return gray


def task2(im, method, threshold, max_inten=255):
    gray = im.copy()
    gray = method(gray, threshold, max_inten)
    plt.figure(figsize=(10, 13))
    plt.imshow(gray, cmap='gray')


def apply_threshold(im, threshold, max_inten=255):
    idx = im > threshold
    im[idx] = max_inten
    im[np.logical_not(idx)] = 0
    return im


def ordered_threshold(size):
    if size == 2:
        return np.array([[0, 2],[3, 1]])
    else:
        size = math.ceil(size / 2)
        U_n = np.ones((size, size))
        top = np.hstack((4*ordered_threshold(size), 4*ordered_threshold(size)+ 2*U_n))
        bottom = np.hstack((4*ordered_threshold(size) + 3*U_n,  4*ordered_threshold(size) + U_n))
        return np.vstack([top, bottom])


def forward_raw_process(im, threshold, raw, errors, max_inten):
    raws = im[raw : raw + errors.shape[0]]
    
    for i in range(1, raws.shape[1]-1):
        error = raws[0, i] - float(raws[0, i] > threshold) * max_inten
        raws[0, i] = float(raws[0, i] > threshold) * max_inten
        matrix_error = error * errors
        start = math.ceil(errors.shape[1] / 2) - 1
        stop = math.floor(errors.shape[1] / 2) + 1
        raws[:, i - start : i + stop] = raws[:, i - start : i + stop] + matrix_error
        raws[raws > max_inten] = max_inten
        raws[raws < 0] = 0

    im[raw : raw + errors.shape[0], :] = raws
    
    return im

def backward_raw_process(im, threshold, raw, errors, max_inten):
    raws = im[raw : raw + errors.shape[0]]
    
    for i in range(raws.shape[1]-2, 0, -1):
        error = raws[0, i] - float(raws[0, i] > threshold) * max_inten
        raws[0, i] = float(raws[0, i] > threshold) * max_inten
        matrix_error = error * errors
        start =  math.floor(errors.shape[1] / 2)
        stop = math.ceil(errors.shape[1] / 2)
        raws[:, i - start : i + stop] = raws[:, i - start : i + stop] + matrix_error 
        raws[raws > max_inten] = max_inten
        raws[raws < 0] = 0
    
    im[raw : raw + errors.shape[0], :] = raws
    
    return im
  
def add_zeros(im):
    img = np.vstack([im, np.zeros((1, im.shape[1]))])
    img = np.hstack([np.zeros((img.shape[0], 1)), img])
    img = np.hstack([img, np.zeros((img.shape[0], 1))])
    
    return img


def error1_method(im, threshold=128, max_inten=255):
    img = im.copy()
    img = add_zeros(img)
    
    for raw in range(im.shape[0]):
        img = forward_raw_process(img, threshold, raw, np.array([[0, 1]]), max_inten)
    
    return img


def error2_method(im, threshold=128, max_inten=255):
    img = im.copy()
    img = add_zeros(img)
    
    for raw in range(im.shape[0]):
        if raw%2 == 0:
            img = forward_raw_process(img, threshold, raw, np.array([[0, 1]]), max_inten)
        if raw%2 == 1:
            img = backward_raw_process(img, threshold, raw, np.array([[1, 0]]), max_inten)
    
    return img


def FS1_method(im, threshold=128, max_inten=255):
    img = im.copy()
    img = add_zeros(img)
    
    for raw in range(im.shape[0]):
        img = forward_raw_process(img, threshold, raw,
                                  np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
                                  max_inten)
    
    return img


def FS2_method(im, threshold=128, max_inten=255):
    img = im.copy()
    img = add_zeros(img)
    
    for raw in range(im.shape[0]):
        if raw%2 == 0:
            img = forward_raw_process(img, threshold, raw,
                                      np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
                                      max_inten)
        if raw%2 == 1:
            img = backward_raw_process(img, threshold, raw,
                                       np.array([[7/16, 0, 0], [1/16, 5/16, 3/16]]),
                                       max_inten)
    
    return img