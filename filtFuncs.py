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

def predict(x,y,R):
  n = len(x)
  x = add(x,[(1-2*random())*R for i in range(n)])
  y = add(y,[(1-2*random())*R for i in range(n)])
  return x,y
