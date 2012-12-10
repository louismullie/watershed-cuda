### Parallel Watershed Segmentation of Medical Images on GPUs

**Contents**

This package contains:

- A CUDA implementation of an order-variant watershed algorithm (`ws_gpu.py` and `kernels.cu`), based on [1-4]. A serial version of the parallel algorithm can be found in `ws_parallel.py`.
- An more efficient serial implementation of the order-variant watershed algorithm (`ws_serial.py`).
- An MPI implementation of a master/slave scheduler that allows distributed computation of the watershed transform (`ws_mpi.py`)
- A set of utilities to read the DICOM format for medical images (heavily inspired from [7]), preprocess the images for the watershed algorithm, and display the results of the watershed transform (`ws_utils.py`).
- A sample DCM image of a thorax CT, obtained from the Cancer Imaging Archive [8] (`test.dcm`).

**Running on one image**

The programs ws_gpu.py, ws_serial.py and ws_parallel.py can be run on an individual DCM image by supplying the image file as an argument. For example,

    python ws_gpu.py test.dcm
    python ws_serial.py test.dcm
  
The result will be displayed in a `matplotlib` popup box.

N.B: the serial implementations of the GPU algorithm (ws_parallel.py) is inherently inefficient and will take several minutes to run on the test image. It is mostly provided as a reference. The ws_serial.py version uses path compression with a reference list for pixel labelling and is much more efficient.

**Running on several images**
The program ws_mpi.py can be run on multiple DCM images by supplying the input directory (with a trailing slash) as a parameter. For example, 

    # Download a sample DCM data set.
    wget louismullie.com/dump/dcm.zip
    mkdir data && mv dcm.zip data/
    cd data && unzip dcm.zip && cd ..
  
    # Run the distributed computation.
    python ws_gpu.py data/

The resulting images will be output as PNG files in the supplied directory. The images that will be saved to disk correspond to the superposition of the watershed lines with the original frame.

**References**

[[1]](http://www.fem.unicamp.br/~labaki/Academic/cilamce2009/1820-1136-1-RV.pdf) Vitor B, Körbes A. Fast image segmentation by watershed transform on graphical hardware.

[[2]](http://www.lbd.dcc.ufmg.br/colecoes/wvc/2009/0012.pdf) Körbes A et al. 2009. A proposal for a parallel watershed transform algorithm for real-time segmentation. Proceedings of Workshop de Visão Computacional WVC’2009.

[[3]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2010. Analysis of a step-by-step watershed algorithm using CUDA. International Journal of Natural Computing Research. 1:16-28.

[[4]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2011. Advances on Watershed Processing on GPU Architectures. ISMM 2011, LNCS 6671, pp. 260–271, 2011.

[[5]](http://code.google.com/p/pydicom/source/browse/source/dicom/contrib/pydicom_Tkinter.py?r=f2c30464fd3b7e553af910ee5a9f5bcf4b3f4ccf) Reference for DICOM reader script.

[[6]](http://cancerimagingarchive.net/) Cancer Imaging Archive.