import numpy as np; import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit
from scipy import ndimage
from skimage.filter import canny

kernel_macros = \
"""
#define INDEX(j,i,ld) ((j) * ld + (i))
#define INF 9999999999
#define PLATEAU 0
#define BLOCK_SIZE 6
// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / BLOCK_SIZE) * (BLOCK_SIZE - 2)-1+(off))
"""

kernel_source = kernel_macros + \
"""

texture<float,2> img;
__constant__ int N_xs[8] = {-1,0,1,1,1,0,-1,-1};
__constant__ int N_ys[8] = {-1,-1,-1,0,1,1,1,0};

__global__ void descent_kernel(float* labeled, const int w, const int h)
{
  int tx = threadIdx.x;  int ty = threadIdx.y;
  int bx = blockIdx.x;   int by = blockIdx.y;
  int bdx = blockDim.x;  int bdy = blockDim.y;
  int i = bdx * bx + tx; int j = bdy * by + ty;
 
  __shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int new_w = w + w * 2;
  int new_h = h + h * 2;
  int p = INDEX(img_y,img_x,w);

  int ghost = (tx == 0 || ty == 0 || 
  tx == bdx - 1 || ty == bdy - 1);
 
  if (bx == bdx - 1 && w % size != 0) {
    //bdx = (size - w % size) + 2;
  }

  if (by == bdy - 1 && w % size != 0) {
    //bdy = (size - h % size) + 2;
  }

  if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
     (bx == (w / size - 1) && tx == bdx - 1) ||
     (by == (h / size - 1) && ty == bdy - 1)) {
       s_I[INDEX(ty,tx,BLOCK_SIZE)] = INF;
  } else {
     s_I[INDEX(ty,tx,BLOCK_SIZE)] = tex2D(img,img_x,img_y);
  }

  __syncthreads();

  if (j < new_h && i < new_w && ghost == 0) {
    float I_q_min = INF;
    float I_p = tex2D(img,img_x,img_y);
  
    int exists_q = 0;

    for (int k = 0; k < 8; k++) {
      int n_x = N_xs[k]+tx; int n_y = N_ys[k]+ty;
      float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];
      if (I_q < I_q_min) I_q_min = I_q;
    }
    
    for (int k = 0; k < 8; k++) {
      int x = N_xs[k]; int y = N_ys[k];
      int n_x = x+tx; int n_y = y+ty;
      int n_tx = L2I(i,n_x); int n_ty = L2I(j,n_y);
      float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];
      int q = INDEX(n_ty,n_tx,w);
      if (I_q < I_p && I_q == I_q_min) {
        labeled[p] = -q;
        exists_q = 1; break;
      }
    }
    if (exists_q == 0) labeled[p] = PLATEAU;
  }

}

__global__ void plateau_kernel(float* L, int* C, const int w, const int h)
{
  int tx = threadIdx.x;  int ty = threadIdx.y;
  int bx = blockIdx.x;   int by = blockIdx.y;
  int bdx = blockDim.x;  int bdy = blockDim.y;
  int i = bdx * bx + tx; int j = bdy * by + ty;
 
  __shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int true_p = INDEX(img_y,img_x,w);
  int p = INDEX(ty,tx,BLOCK_SIZE);
  int new_w = w + w * 2;
  int new_h = h + h * 2;
  int ghost = (tx == 0 || ty == 0 || 
  tx == bdx - 1 || ty == bdy - 1);

  // Load data into shared memory.
  if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
     (bx == (w / size - 1) && tx == bdx - 1) ||
     (by == (h / size - 1) && ty == bdy - 1)) {
       s_L[INDEX(ty,tx,BLOCK_SIZE)] = INF;
  } else {
     s_L[INDEX(ty,tx,BLOCK_SIZE)] =
     L[INDEX(img_y,img_x,w)];
  }

  __syncthreads();

  if (j < new_h && i < new_w && 
    s_L[p] == PLATEAU && ghost == 0) {
    float I_p = tex2D(img,img_x,img_y); 
    float I_q;
    int n_x, n_y; float L_q;

    for (int k = 0; k < 8; k++) {
      n_x = N_xs[k]+tx; n_y = N_ys[k]+ty;
      L_q = s_L[INDEX(n_y,n_x,BLOCK_SIZE)];
      if (L_q == INF || L_q >= 0) continue;
      int n_tx = L2I(i,n_x); int n_ty = L2I(j,n_y);
      int q = INDEX(n_ty,n_tx,w);
      I_q = tex2D(img,n_tx,n_ty);
      if (I_q == I_p && L[true_p] != -q) {
        L[true_p] = -q; 
        atomicAdd(&C[0], 1); 
        break;
      }
    }
  }

}

__global__ void increment_kernel(float* L, const int w, const int h)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int p = INDEX(j,i,w);

  if (j < h && i < w && L[p] == PLATEAU) {
    L[p] = p + 1;
  }
}

__global__ void minima_kernel(float* L, int* C, const int w, const int h)
{
  int tx = threadIdx.x;  int ty = threadIdx.y;
  int bx = blockIdx.x;   int by = blockIdx.y;
  int bdx = blockDim.x;  int bdy = blockDim.y;
  int i = bdx * bx + tx; int j = bdy * by + ty;
 
  __shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int true_p = INDEX(img_y,img_x,w);
  int s_p = INDEX(ty,tx,BLOCK_SIZE);
  int new_w = w + w * 2;
  int new_h = h + h * 2;
  int ghost =  (tx == 0 || ty == 0 || 
  tx == bdx - 1 || ty == bdy - 1) ? 1 : 0;

  // Load data into shared memory.
  if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
     (bx == (w / size - 1) && tx == bdx - 1) ||
     (by == (h / size - 1) && ty == bdy - 1)) {
     s_L[INDEX(ty,tx,BLOCK_SIZE)] = INF;
  } else {
    s_L[s_p] = L[INDEX(img_y,img_x,w)];
  }

  __syncthreads();

  int active = (j < new_h && i < 
  new_w && s_L[s_p] > 0) ? 1 : 0;

  if (active == 1 && ghost == 0) {
    for (int k = 0; k < 8; k++) {
      int n_x = N_xs[k] + tx; int n_y = N_ys[k] + ty;
      int s_q = INDEX(n_y,n_x,BLOCK_SIZE);
      if (s_L[s_q] == INF) continue;
      if (s_L[s_q] > s_L[s_p])
        s_L[s_p] = s_L[s_q];
    }
    if (L[true_p] != s_L[s_p]) {
      L[true_p] = s_L[s_p];
      atomicAdd(&C[0],1);
    }
  }
}

__global__ void flood_kernel(float* L, int* C, const int w, const int h)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int p = INDEX(j,i,w); int q;

  if (j < h && i < w && L[p] <= 0) {
    q = -L[p];
    if (L[q] > 0 && L[p] != L[q]) {
      L[p] = L[q];
      atomicAdd(&C[0],1);
    }
  }
}

"""

print "Compiling CUDA kernels..."
main_module = nvcc.SourceModule(kernel_source)
descent_kernel = main_module.get_function("descent_kernel")
image_texture = main_module.get_texref("img")
plateau_kernel = main_module.get_function("plateau_kernel")
minima_kernel = main_module.get_function("minima_kernel")
flood_kernel = main_module.get_function("flood_kernel")
increment_kernel = main_module.get_function("increment_kernel")

def watershedGPU(I):

  # Get contiguous image + shape.
  height, width = I.shape
  I = np.float32(I.copy())

  # Get block/grid size for steps 1-3.
  block_size =  (16,16,1)
  grid_size =   (width/(block_size[0]-2),
                height/(block_size[0]-2))

  # Get block/grid size for step 4.
  block_size2 = (16,16,1)
  grid_size2  = (width/(block_size2[0]-2),
                height/(block_size2[0]-2))

  # Initialize variables.
  labeled       = np.zeros([height,width]) 
  labeled       = np.float32(labeled)
  width         = np.int32(width)
  height        = np.int32(height)
  count         = np.int32([0])

  # Transfer labels asynchronously.
  labeled_d = gpu.to_gpu_async(labeled)
  counter_d = gpu.to_gpu_async(count)

  # Bind CUDA textures.
  I_cu = cu.matrix_to_array(I, order='C')
  cu.bind_array_to_texref(I_cu, image_texture)

  # Step 1.
  descent_kernel(labeled_d, width, 
  height, block=block_size, grid=grid_size)
  
  start_time = cu.Event()
  end_time = cu.Event()
  start_time.record()

  # Step 2.
  increment_kernel(labeled_d,width,height,
  block=block_size2,grid=grid_size2)
  counters_d = gpu.to_gpu(np.int32([0]))
  old, new = -1, -2
  i = 0
  while i < 174: #old != new:
    old = new
    i += 1
    minima_kernel(labeled_d, counters_d,
    width, height, block=block_size, grid=grid_size)
    new = counters_d.get()[0]
  print i

  # End GPU timers.
  end_time.record()
  end_time.synchronize()
  gpu_time = start_time.\
  time_till(end_time) * 1e-3

  # Step 3.
  counters_d = gpu.to_gpu(np.int32([0]))
  old, new = -1, -2
  while old != new:
    old = new
    plateau_kernel(labeled_d, counters_d, width,
    height, block=block_size, grid=grid_size)
    new = counters_d.get()[0]
  
  # Step 4
  counters_d = gpu.to_gpu(np.int32([0]))
  old, new = -1, -2
  while old != new:
    old = new
    flood_kernel(labeled_d, counters_d, width,
    height, block=block_size2, grid=grid_size2)
    new = counters_d.get()[0]

  result = labeled_d.get()
  
  # Print time.
  print "Total time: %f s" % gpu_time

  return result

def neighbours(j,i):
  return [(j-1,i-1),(j-1,i),(j-1,i+1),(j,i+1),
  (j+1,i+1),(j+1,i),(j+1,i-1),(j,i-1)]

def outside(I,n):
  j,i = n
  h, w = I.shape
  if j < 0 or j > h-1 or \
  i < 0 or i > w-1: return True
  else: return False

def showEdges(L,I):
  I = np.int32(I)
  E = np.zeros_like(I)
  height,width = I.shape
  for j in range(0,height):
    for i in range(0,width):
      p = (j,i); c = 0
      for u in neighbours(j,i):
        if outside(I,u): continue
        if L[u] < L[p] and E[u] != 1: E[p] = 1
  plt.imshow(E,cmap='gray')
  plt.show()

# Read in the DICOM image data.
O = img.imread("test13.png")[:,:,0]
I = np.int32(canny(O,2,0.2,0.2)) * np.max(O)
I = ndimage.gaussian_filter(I,1)
L = watershedGPU(I)
# showEdges(L,O)
