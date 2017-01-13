#!python
from numpy import *
from numpy.random import *
import numpy as np

def systematic_resample(weights):
  n = weights.shape[0]
  w = 1./n
  indices = []
  C = [0.0]
  for i in range(n):
      C.append(C[i] + weights[i])
  j = 0
  u0 = random()*w
  for u in range(n):
    while u0+u*w > C[j]:
      j+=1
    indices.append(j-1)
  return indices

def default(width, height, size=1):
    w1 = np.random.randint(5, 32, size=size)
    x1 = np.random.randint(w1,width-w1, size=size)
    y1 = np.random.randint(w1,height-w1, size=size)
    return x1, y1, w1

def window_is_acceptable(x, y, w, width, height, illegal_windows):
    # Check if the particle is inside the image boundaries
    if w < 5 or w > 64 or x < w or x > width - w or y < w or y > height - w:
        return False
    # Check if the particle collides with an illegal window
    for x1, y1, w1 in illegal_windows:
        if x1 is not None:
            if x + w > x1 - w1 and x - w < x1 + w1 \
                    and y + w > y1 - w1 and y - w < y1 + w1:
                return False
    return True

def predict(x,y,w,R,width,height,illegal_windows):
    n = x.shape[0]
    for i in range(n):
        acceptable = False
        count = 10
        x_old = x[i]
        y_old = y[i]
        w_old = w[i]
        while not acceptable and count > 0:
            count -= 1
            x[i] = (x_old + normal(0.0, R)).astype(int)
            y[i] = (y_old + normal(0.0, R)).astype(int)
            w[i] = (w_old + normal(0.0, R)).astype(int)
            # Resample if the new particle location is not acceptable
            # (is outside image boundaries or collides with another illegal
            # window)
            acceptable = window_is_acceptable(x[i], y[i], w[i], width, height,
                                              illegal_windows)
        if count == 0:
            x[i], y[i], w[i] = default(width, height)
    return x,y,w

def multinomial_resample(weights,num):
  n = weights.shape[0]
  w = 1./n
  indices = []
  C = [0.0]
  for i in range(n):
      C.append(C[i] + weights[i])
  for u in range(num):
    j=0
    u0 = random()
    while u0 > C[j]:
      j+=1
    indices.append(j-1)
  return indices
