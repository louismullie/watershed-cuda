import numpy as np
from sys import argv
from ws_utils import *

# Watershed segmentation as described in:
# 
# Vitor B, Korbes A. Fast image segmentation 
# by watershed transform on graphical hardware.
# ISMM'11 Proceedings of the 10th international 
# conference on Mathematical morphology and its 
# applications to image and signal processing.
# 
# Follows general strategies of minima 
# (connected component) labeling different
# from those in the serial implementation.
def watershed(I):
  # Initialize.
  height, width = I.shape
  L = np.zeros_like(I)

  # * Step 1 * # 
  
  # Find the lowest neighbour of each pixel
  # (direct path of steepest descent).
  for j in range(0, height):
    for i in range(0, width):
      p, N = (j, i), neighbours(j,i)
      I_u_min = INFINITE
      for u in N:
        if outside(I,u): continue
        if I[u] < I_u_min: I_u_min = I[u]
      exists_q = False
      for q in N:
        if outside(I,q): continue
        if I[q] < I[p] and I[q] == I_u_min:
          L[p] = -t2i(q,width)
          exists_q = True; break
      if not exists_q: L[p] = PLATEAU
  show_progress()
  
  # * Step 2 * # 
  
  # Find the nearest border of internal pixels 
  # of plateaus, propagating uniformly from the borders.
  for j in xrange(0, height):
    for i in xrange(0, width):
      if L[j,i] == PLATEAU:
        L[j,i] = t2i((j,i),width) + 1
  show_progress()

  stable = False
  while not stable:
    stable = True
    # Propagate minimal index.
    for j in xrange(0, height):
      for i in xrange(0, width):
        p, N = (j,i), neighbours(j,i)
        if L[p] <= 0: continue
        for q in N:
          if outside(I,q): continue
          if L[q] > L[p]:
            L[p] = L[q]
            stable = False
    show_progress()

  # * Step 3 * # 
  
  # Minima labelling by maximal neighbour address.
  stable = False
  while not stable:
    tL, stable = L.copy(), True
    for j in xrange(0, height):
      for i in xrange(0, width):
        p, N = (j,i), neighbours(j,i)
        if L[p] != PLATEAU: continue
        for q in N:
          if outside(I,q): continue
          if L[q] < 0 and I[q] == I[p]:
            tL[p] = -q; stable = False; break
    show_progress()
    L = tL.copy()

  # * Step 4 * # 
  
  # Label all pixels by flooding from minima.
  stable = False
  while not stable:
    stable = True
    for j in xrange(0,height):
      for i in xrange(0,width):
        if L[j,i] > 0: continue
        q = -L[j,i]
        if L[i2t(q,width)] > 0:
          L[j,i] = L[i2t(q,width)]
          stable = False
    show_progress()
  
  # Return the labeled image.
  return L

if __name__ == '__main__':
  # Show the usage information.
  if len(argv) != 2:
    print "Usage: python ws_parallel.py test_image.dcm"
    exit()
  # Read in the DICOM image data.
  O = read_dcm(argv[1])
  # Preprocess the image.
  I = preprocess(O)
  # Get the watershed transform.
  L = watershed(I)
  # Show the final edges.
  showEdges(L,O)