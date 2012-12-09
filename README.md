<<<<<<< HEAD
### Parallel Watershed Segmentation using CUDA

This repo contains:

- An implementation of serial order-invariant toboggan-based watershed segmentation as described in [1].
- Serial implementations of two different order-variant parallel toboggan watershed algorithms proposed in [2-5]
- A CUDA implementation of the order-variant watershed algorithm as described in [5], using PyCUDA wrappers.

See [6] for a recent and comprehensive review of watershed segmentation techniques.

[[1]](http://www.ncbi.nlm.nih.gov/pubmed/16519350) Lin Y, Tsai Y, Hung Y, Shih Z. 2006. Comparison between immersion-based and toboggan-based watershed image segmentation. IEEE Transactions on Image Processing, vol. 15, n. 3, pp. 632–640.

[[2]](http://www.fem.unicamp.br/~labaki/Academic/cilamce2009/1820-1136-1-RV.pdf) Vitor B, Körbes A. Fast image segmentation by watershed transform on graphical hardware.

[[3]](http://www.lbd.dcc.ufmg.br/colecoes/wvc/2009/0012.pdf) Körbes A et al. 2009. A proposal for a parallel watershed transform algorithm for real-time segmentation. Proceedings of Workshop de Visão Computacional WVC’2009.

[[4]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2010. Analysis of a step-by-step watershed algorithm using CUDA. International Journal of Natural Computing Research. 1:16-28.

[[5]](http://parati.dca.fee.unicamp.br/media/Attachments/courseIA366F2S2010/aula10/ijncr.pdf) Körbes A et al. 2011. Advances on Watershed Processing on GPU Architectures. ISMM 2011, LNCS 6671, pp. 260–271, 2011.

[[6]](http://cscjournals.org/csc/manuscript/Journals/IJIP/volume5/Issue5/IJIP-409.pdf) Mahmoudi R, Akil M. 2011. Analyses of the Watershed Transform. International Journal of Image Processing, Vol. 5 no. 5, p. 521-541.
=======
File listing:

ws_gpu.py |_g
>>>>>>> Finalizing implementation for class.
