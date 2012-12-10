import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit
from sys import argv
from ws_utils import *

# Read and compile CUDA kernels.
print "Compiling CUDA kernels..."
kernel_source = open("kernels.cu").read()
main_module = nvcc.SourceModule(kernel_source)
descent_kernel = main_module.get_function("descent_kernel")
image_texture = main_module.get_texref("img")
plateau_kernel = main_module.get_function("plateau_kernel")
minima_kernel = main_module.get_function("minima_kernel")
flood_kernel = main_module.get_function("flood_kernel")
increment_kernel = main_module.get_function("increment_kernel")

# PyCUDA wrapper for watershed.
def watershed(I):

  # Get contiguous image + shape.
  height, width = I.shape
  I = np.float32(I.copy())

  # Get block/grid size for steps 1-3.
  block_size =  (6,6,1)
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

  while old != new:
    old = new
    minima_kernel(labeled_d, counters_d,
    width, height, block=block_size, grid=grid_size)
    new = counters_d.get()[0]

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
  
  # End GPU timers.
  end_time.record()
  end_time.synchronize()
  gpu_time = start_time.\
  time_till(end_time) * 1e-3

  # print str(gpu_time)

  return result

if __name__ == '__main__':
  # Show the usage information.
  if len(argv) != 2:
    print "Usage: python ws_gpu.py test_image.dcm"
  # Read in the DICOM image data.
  O = read_dcm(argv[1])
  # Preprocess the image.
  I = preprocess(O)
  # Get the watershed transform.
  L = watershed(I)
  # Show the final edges.
  showEdges(L,O)
