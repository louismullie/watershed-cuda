import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import Queue
INFINITE = 9999999

# Mark non-assigned pixels.
MASK = -1
# Mark watershed ridges.
WSHED = -2
IMAGE_WIDTH = 6

def index(ind):
  return ind[0] * IMAGE_WIDTH + ind[1]

def union(a,b):
  return list(set(a) | set(b))

def neighbours(p):
  j,i = p
  return [(j-1,i-1), (j-1,i), (j-1,i+1), (j,i+1),
   (j+1,i+1), (j+1,i), (j+1,i-1), (j,i-1)]

def outside(I,neighbour):
  j, i = neighbour
  height, width = I.shape
  if j < 0 or j > height-1 or \
  i < 0 or i > width-1: return True
  else: return False


def unique(S):
  return len(S) > 0 and all(p == S[0] for p in S)

def resolve2(p,L,sliding):
  if L[p] == MASK:
    S = sliding[p]
    for q in S:
      L = resolve(q,L,sliding)
    if unique(S):
      L[p] = index(S[0])
    else:
      L[p] = WSHED
  return L

def resolve(p,L,sliding,dist):
  if L[p] == MASK:
    S = sliding[p]
    for q in S:
      L = resolve(q,L,sliding,dist)
    d_min = INFINITE
    for q in S:
      if dist[q] < d_min:
        d_min = dist[q]
    S_min = []
    for q in S:
      if dist[q] == d_min:
        S_min.append(q)
    if unique(S_min):
      L[p] = index(S[0])
    else:
      L[p] = WSHED
  return L
  
I = [[14, 14, 10, 8, 0, 0],
    [14, 14, 12, 8, 3, 5],
    [24, 25, 25, 14, 14, 8],
    [14, 16, 24, 23, 14, 14],
    [12, 13, 16, 25, 24, 24],
    [10, 12, 14, 24, 24, 24]]

I = np.array(I)

# I = img.imread("test.png")[:,:,0]
height,width = I.shape

# Initialize.
L = np.zeros_like(I)
L.fill(MASK)
Q = Queue.Queue()
sliding = dict()
dist = dict()

# Start timer.
start = time.time()

# Simulation of sliding for all C1 pixels.
for j in range(0,height):
  for i in range(0,width):
    p = (j,i); h = I[p]
    N = neighbours(p)
    h_min = INFINITE
    for q in N:
      if outside(I,q): continue
      if I[q] < h_min: h_min = I[q]
    if h > h_min:
      S = []
      for q in N:
        if outside(I,q): continue
        if I[q] == h_min:
          S.append(q)
      sliding[p] = S
      Q.put(p)
      dist[p] = 0
    else: sliding[p] = []

# Simulation of keep-sliding for all C2 pixels.
while not Q.empty():
  p = Q.get()
  d = dist[p] + 1
  h = I[p]
  N = neighbours(p)
  for q in N:
    if outside(I,q): continue
    if I[q] != h: continue
    if not sliding[q]:
      sliding[q].append(p)
      dist[q] = d
      Q.put(q)
    elif dist[q] == d:
      sliding[q].append(p)

# Number of basins (serves
# to label new basins).
basins = 1

# Labeling for C3 pixels.
for j in range(0,height):
  for i in range(0,width):
    p0 = (j,i)
    if sliding[p0]: continue
    if L[p0] != MASK: continue
    L[p0] = basins
    basins += 1
    Q.put(p0)
    h = I[p0]
    while not Q.empty():
      p = Q.get()
      N = neighbours(p)
      for q in N:
        if outside(I,q): continue
        if I[q] != h: continue
        if L[q] == MASK:
          L[q] = L[p0]
          Q.put(q)

# Tobogganing (depth first search)
for j in range(0,height):
  for i in range(0,width):
    L = resolve((j,i),L,sliding,dist)

print L
# plt.imshow(L,cmap='jet')
# plt.show()
end = time.time()

print "Time: %f s" % (end-start)