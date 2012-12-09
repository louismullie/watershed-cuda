import numpy as np
from sys import argv
from ws_utils import *

# Watershed segmentation as described in:
#
# Korbes A et al. 2010. Analysis of a step-by-step 
# watershed algorithm using CUDA. International 
# Journal of Natural Computing Research. 1:16-28.
# 
# Adapted from Lin Y, Tsai Y, Hung Y, Shih Z. 
# 2006. Comparison between immersion-based and 
# toboggan-based watershed image segmentation. 
# IEEE Transactions on Image Processing, vol. 15, 
# n. 3. pp. 632-640.
# 
# Uses a labelling algorithm of based on a reference 
# list for path compression and representative propagation.
def watershed(I):
  
  # Initialize.
  height,width = I.shape
  L = np.zeros_like(I)

  # * Step 1 * # 
  
  # Find the lowest neighbour of each pixel
  # (direct path of steepest descent).
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
          L[p] = -t2i(q,width)
          exists_q = True; break
      if not exists_q: L[p] = PLATEAU
  show_progress()
  
  # * Step 2 * # 
  
  # Find the nearest border of internal pixels 
  # of plateaus, propagating uniformly from the borders.
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
            if tL[p] != -t2i(q,width):
              stable = False
            tL[p] = -t2i(q,width)
            break
    L = tL.copy()
    show_progress()

  for j in range(0, height):
    for i in range(0, width):
      p = (j,i)
      if L[p] == PLATEAU:
        L[p] = t2i(p,width)

  # * Step 3 * # 

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
          if L[i2t(L[p],width)] != q:
            stable = False
          L[i2t(L[p],width)] = q
    # Representative propagation.
    for j in range(0, height):
      for i in range(0, width):
        p = (j,i)
        if L[p] <= PLATEAU: continue
        label = L[p]
        if label != t2i(p,width):
          ref = None
          while label != ref:
            ref = label
            label = L[i2t(ref,width)]
          if L[p] != label:
            stable = False
          L[p] = label
    show_progress()

  # * Step 4 * # 
  
  # Pixel labeling by path compression.
  L = np.abs(L)
  for j in range(0,height):
    for i in range(0,width):
      p = (j,i)
      if outside(I,p): continue
      label = L[p]
      if label != t2i(p,width):
        ref = None
        while label != ref:
          ref = label
          label = L[i2t(ref,width)]
        L[p] = label
  show_progress()
  
  # Return the labeled image.
  return L

if __name__ == '__main__':
  # Show the usage information.
  if len(argv) != 2:
    print "Usage: python ws_serial.py test_image.dcm"
    exit()
  # Read in the DICOM image data.
  O = read_dcm(argv[1])
  # Preprocess the image.
  I = preprocess(O)
  # Get the watershed transform.
  L = watershed(I)
  # Show the final edges.
  showEdges(L,O)