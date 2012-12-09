import numpy as np; import os
from skimage.filter import canny
from scipy import ndimage
import matplotlib.image as img
import matplotlib.pyplot as plt
import sys; import Queue


# Constants.
PLATEAU = 0
INFINITE = 9e10

# Gets the neighbours of (j,i)
def neighbours(j,i):
  return [(j-1,i-1),(j-1,i),(j-1,i+1),(j,i+1),
  (j+1,i+1),(j+1,i),(j+1,i-1),(j,i-1)]

# Determines if tuple n is outside
# of the boundaries of the image I.
def outside(I,n):
  j,i = n
  h, w = I.shape
  if j < 0 or j > h-1 or \
  i < 0 or i > w-1: return True
  else: return False

# Check if array contains unique element.
def unique(S):
  return len(S) > 0 and all(p == S[0] for p in S)

# Convert tuple to index.
def t2i(tup,width):
  return tup[0] * width + tup[1]

# Convert index to tuple.
def i2t(ind,width):
  return (ind / width, ind % width)

# Show edges in the final watershed.
def showEdges(L,I):
  E = getEdges(L)
  plt.imshow(E+I,cmap='gray')
  plt.show()

# Get edges from watershed image
# (i.e. watershed lines).
def getEdges(L):
  E = np.zeros_like(L)
  height,width = L.shape
  for j in range(0,height):
    for i in range(0,width):
      p = (j,i); c = 0
      for u in neighbours(j,i):
        if outside(L,u): continue
        if L[u] < L[p] and E[u] != 1: E[p] = 1
  return E

# Preprocess an image with a canny filter
# followed by a 4D gaussian filter.
def preprocess(I):
  I = np.int32(canny(I,2,0.2,0.2)) * np.max(I)
  I = ndimage.gaussian_filter(I,1)
  return I

# Show progress dots.
def show_progress():
  sys.stdout.write(".")
  sys.stdout.flush()

# Heavily inspired from http://code.google.com/p/pydicom/
# source/browse/source/dicom/contrib/pydicom_Tkinter.py?
# r=f2c30464fd3b7e553af910ee5a9f5bcf4b3f4ccf
def pgm_from_np(arr, window_center, window_width, lut_min=0, lut_max=255):

    # Basic sanity checking.
    if np.isreal(arr).sum() != arr.size: raise ValueError
    if lut_max != 255: raise ValueError
    if arr.dtype != np.float64: arr = arr.astype(np.float64)

    # Get window information.
    window_width = max(1, window_width)
    wc, ww = np.float64(window_center), np.float64(window_width)
    lut_range = np.float64(lut_max) - lut_min

    # Transform the image.
    minval = wc - 0.5 - (ww - 1.0) / 2.0
    maxval = wc - 0.5 + (ww - 1.0) / 2.0
    min_mask = (minval >= arr)
    to_scale = (arr > minval) & (arr < maxval)
    max_mask = (arr >= maxval)
    if min_mask.any(): arr[min_mask] = lut_min

    # Scale the image to the right proportions.
    if to_scale.any(): arr[to_scale] = \
      ((arr[to_scale] - (wc - 0.5)) /
      (ww - 1.0) + 0.5) * lut_range + lut_min
    if max_mask.any(): arr[max_mask] = lut_max

    arr = np.rint(arr).astype(np.uint8)

    return arr

# Read in a DICOM file.
def read_dcm(file_name):
  data = dicom.read_file(file_name)
  arr = data.pixel_array.astype(np.float64)
  # Rescale image.
  if ('RescaleIntercept' in data) and ('RescaleSlope' in data):
    intercept = int(data.RescaleIntercept)
    slope = int(data.RescaleSlope)
    arr = slope * arr + intercept
  wc = (arr.max() + arr.min()) / 2.0
  ww = arr.max() - arr.min() + 1.0
  if ('WindowCenter' in data) and ('WindowWidth' in data):
    wc = data.WindowCenter
    ww = data.WindowWidth
    try: wc = wc[0]
    except: pass
    try: ww = ww[0]
    except: pass
  return pgm_from_np(arr, wc, ww)
  
def strip_extension(path):
  return os.path.splitext(path)[0]