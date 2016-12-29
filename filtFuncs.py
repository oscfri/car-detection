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

def predict(x,y,w,R):
  n = x.shape[0]
  w = (w + normal(0.0, R, n)).astype(int)
  x = (x + normal(0.0, R, n)).astype(int)
  y = (y + normal(0.0, R, n)).astype(int)
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
