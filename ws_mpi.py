from sys import argv
import numpy as np
from mpi4py import MPI
from collections import deque
from ws_utils import * 
import os

# Flags for MPI.
EXECUTE = 0
TERMINATE = 1

# Read a DCM image from disk,
# process it and write the result
# as a PNG file in the same dir.
def process_image(comm):
  status = MPI.Status()
  while True:
    msg = comm.recv(None, 0, MPI.ANY_TAG, status)
    if status.Get_tag() == TERMINATE: break
    folder,file_name = msg
    f = folder + file_name
    # Read in image.
    O = read_dcm(f)
    # Preprocess image.
    I = preprocess(O)
    # Perform watershed.
    L = watershed(I)
    # Get watershed lines.
    E = getEdges(L,O)
    # Show progress dots.
    show_progress()
    # Save image to disk
    out = strip_extension(f) + ".png"
    img.imsave(out, E, cmap='gray')
    comm.send(None, 0)

# Distribute images across the
# available MPI processes.
def distribute_images(comm,folder):

  size = comm.Get_size()
  status = MPI.Status()
  queue = os.listdir(folder)

  # Send out initial data.
  for i in range(1,size):
    file_name = queue.pop(0)
    msg = [folder,file_name]
    comm.send(msg,i)

  # Process all remaining.
  while queue:
    file_name = queue.pop(0)
    msg = [folder,file_name]
    comm.recv(None, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
    comm.send(msg, status.Get_source(), EXECUTE)

  # Terminate all workers.
  for i in range (1,size):
    comm.send(None,i,TERMINATE)

# If running as main script:
if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  # Show usage.
  if len(argv) != 2:
    if rank == 0:
      print "Usage: mpirun -N [processes] ws_mpi.py [folder]."
      print "Folder must contain DCM images only."
    exit()
  # Run as master.
  if rank == 0:
    start_time = MPI.Wtime()
    distribute_images(comm,argv[1])
    end_time = MPI.Wtime()
    print "Total time: %f" % \
    (end_time - start_time)
  # Run as slave.
  else:
    from ws_gpu import watershed
    process_image(comm)
