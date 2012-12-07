# Fast image segmentation by watershed
# transform on graphical hardware.
# Follows general strategies of minima 
# (connected component) labeling found in 
# algorithms such as Bieniek and Moga (1998), 
# Osma-Ruiz et al. (2007) and Lin et al. (2006).
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filter import canny

PLATEAU = 0
INFINITE = 9999999999

# Index to tuple
def i2t(ind):
  return (ind / IMAGE_WIDTH, ind % IMAGE_WIDTH)

def t2i(ind):
  return ind[0] * IMAGE_WIDTH + ind[1]

# Convert 2D-index to 1D-index.
def index(ind):
  return ind[0] * IMAGE_WIDTH + ind[1]

# Find the lowest neighbour of each pixel
# (direct path of steepest descent).
def findDescents(I, L):
  height, width = I.shape
  for j in range(0, height):
    for i in range(0, width):
      p, N = (j, i), neighbours2(j,i)
      I_u_min = INFINITE
      for u in N:
        if outside(I,u): continue
        if I[u] < I_u_min: I_u_min = I[u]
      exists_q = False
      for q in N:
        if outside(I,q): continue
        if I[q] < I[p] and I[q] == I_u_min:
          L[p] = -index(q)
          exists_q = True; break
      if not exists_q: L[p] = PLATEAU
  return L

# Label minima by maximal neighbour address.
def labelMinima(I, L):
  height, width = I.shape
  for j in xrange(0, height):
    for i in xrange(0, width):
      if L[j,i] == PLATEAU:
        L[j,i] = index((j,i)) + 1
  stable = False
  while not stable:
    print "Step 2"
    stable = True
    # Propagate minimal index.
    for j in xrange(0, height):
      for i in xrange(0, width):
        p, N = (j,i), neighbours2(j,i)
        if L[p] <= 0: continue
        for q in N:
          if outside(I,q): continue
          if L[q] > L[p]:
            L[p] = L[q]
            stable = False
  return L

# Find the nearest border of internal pixels
# of plateaus, propagating uniformly from the 
# borders.
def findPlateaus(I, L):
  height, width = I.shape
  stable = False
  print "Step 3"
  while not stable:
    tL, stable = L.copy(), True
    for j in xrange(0, height):
      for i in xrange(0, width):
        p, N = (j,i), neighbours2(j,i)
        if L[p] != PLATEAU: continue
        for q in N:
          if outside(I,q): continue
          if L[q] < 0 and I[q] == I[p]:
            tL[p] = -q; stable = False; break
    L = tL.copy()
  return L

# Label all pixels by flooding from minima.
def floodRegions(I, L):
  height, width = I.shape
  stable = False
  print "Step 4"
  while not stable:
    stable = True
    for j in xrange(0,height):
      for i in xrange(0,width):
        if L[j,i] > 0: continue
        q = -L[j,i]
        if L[i2t(q)] > 0:
          L[j,i] = L[i2t(q)]
          stable = False
  return L

def watershedTransform(I):
  height, width = I.shape
  global IMAGE_WIDTH
  IMAGE_WIDTH = width
  L = np.zeros([height,width])
  L = findDescents(I, L)
  L = labelMinima(I, L)
  L = findPlateaus(I, L)
  L = floodRegions(I, L) 
  return L

# Helper methods.
def neighbours2(j,i):
  return [(j-1,i-1), (j-1,i), (j-1,i+1), (j,i+1),
   (j+1,i+1), (j+1,i), (j+1,i-1), (j,i-1)]

def outside(I,neighbour):
  j, i = neighbour
  height, width = I.shape
  if j < 0 or j > height-1 or \
  i < 0 or i > width-1: return True
  else: return False
    
def showEdges(L, I):
  E = np.zeros_like(I)
  height,width = I.shape
  for j in range(0,height):
    for i in range(0,width):
      p = (j,i); c = 0
      for u in neighbours2(j,i):
        if outside(I,u): continue
        if L[u] < L[p] and E[u] != 1: E[p] = 1
  plt.imshow(I + E,cmap='gray')
  plt.show()

O = img.imread("test.png")[:,:,0]
I = np.int32(canny(O, 2, 0.2, 0.2)) * np.max(O)
I = ndimage.gaussian_filter(I, 1)
L = watershedTransform(I)
showEdges(L,O)
print "Done."