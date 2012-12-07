# Analysis of a step-based watershed algorithm using CUDA.
# Uses the labelling algorithm of K.A.Hawick et al. (2009), 
# based on a reference list for path compression and 
# representative propagation.
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

# Constants.
PLATEAU = 0
INFINITE = 99999
WIDTH = 32

# Helper methods.
def neighbours(j,i):
  return [(j-1,i-1), (j-1,i), (j-1,i+1), (j,i+1),
   (j+1,i+1), (j+1,i), (j+1,i-1), (j,i-1)]

def outside(I,neighbour):
  j, i = neighbour
  height, width = I.shape
  if j < 0 or j > height-1 or \
  i < 0 or i > width-1: return True
  else: return False

def index(ind):
  return ind[0] * WIDTH + ind[1]

def i2t(ind):
  return (ind / WIDTH, ind % WIDTH)

# Test array.
I = [[14, 14, 10, 8, 0, 0],
   [14, 14, 12, 8, 3, 5],
   [24, 25, 25, 14, 14, 8],
   [14, 16, 24, 23, 14, 14],
   [12, 13, 16, 25, 24, 24],
   [10, 12, 14, 24, 24, 24]]

I = np.array(I)
I = img.imread("test2.png")[:,:,0]
L = np.zeros_like(I)

height, width = I.shape

# Step 1.
for j in range(0, height):
  for i in range(0, width):
    p = (j, i)
    I_u_min = INFINITE
    for u in neighbours(j,i):
      if outside(I,u): continue
      if I[u] < I_u_min: I_u_min = I[u]
    exists_q = False
    for q in neighbours(j,i):
      if outside(I,q): continue
      if I[q] < I[p] and I[q] == I_u_min:
        L[p] = -index(q)
        exists_q = True; break
    if not exists_q: L[p] = PLATEAU

print L


# Step 2.
stable = False
while not stable:
  stable = True
  tL = L.copy()
  for j in range(0, height):
    for i in range(0, width):
      p = (j, i)
      if L[p] != PLATEAU: continue
      for q in neighbours(j,i):
        if outside(I,q): continue
        if L[q] < 0 and I[q] == I[p]:
          if tL[p] != -index(q):
            stable = False
          tL[p] = -index(q)
          break
  L = tL.copy()

for j in range(0, height):
  for i in range(0, width):
    p = (j,i)
    if L[p] == PLATEAU:
      L[p] = index(p)

# Step 3.
stable = False
while not stable:
  stable = True
  # Propagation of minimal index.
  for j in range(0, height):
    for i in range(0, width):
      p = (j,i)
      if L[p] <= PLATEAU: continue
      q = INFINITE
      for u in neighbours(j,i):
        if outside(I,u): continue
        if I[u] == I[p] and L[u] < q:
          q = L[u]
      if q < L[p]:
        if L[i2t(L[p])] != q:
          stable = False
        L[i2t(L[p])] = q
  # Representative propagation/path compressing.
  for j in range(0, height):
    for i in range(0, width):
      p = (j,i)
      if L[p] <= PLATEAU: continue
      label = L[p]
      if label != index(p):
        ref = None
        while label != ref:
          ref = label
          label = L[i2t(ref)]
        if L[p] != label:
          stable = False
        L[p] = label

# Important
L = np.abs(L)

for j in range(0,height):
  for i in range(0,width):
    p = (j,i)
    if outside(I,p): continue
    label = L[p]
    if label != index(p):
      ref = None
      while label != ref:
        ref = label
        label = L[i2t(ref)]
      L[p] = label
  

plt.imshow(L,cmap='jet',interpolation='nearest')
plt.show()


