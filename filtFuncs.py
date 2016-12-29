#!python
from numpy import *
from numpy.random import *

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

def predict(x, y, w, R):
  n = x.shape[0]
  x = (x + multiply(normal(0.0, 1.0, n), w)).astype(int)
  y = (y + multiply(normal(0.0, 1.0, n), w)).astype(int)
  w = (w + multiply(normal(0.0, 4.0, n), w)).astype(int)
  return x, y, w
