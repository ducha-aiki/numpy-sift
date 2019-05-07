This is an python implementation of SIFT patch descriptor. 
It is derived from Michal Perdoch C++ implementation at https://github.com/perdoch/hesaff

The SIFT descriptor code is protected under a US Patent 6,711,293. A
license MUST be obtained from the University of British Columbia for
use of SIFT code, files numpy_sift.py, in commercial
applications (see LICENSE.SIFT for details)


There are different implementations of the SIFT on the web. I tried to match [Michal Perdoch implementation](https://github.com/perdoch/hesaff/blob/master/siftdesc.cpp), which gives high quality features for image retrieval [CVPR2009](http://cmp.felk.cvut.cz/~chum/papers/perdoch-cvpr09.pdf). However, on planar datasets, it is inferior to [vlfeat implementation](http://www.vlfeat.org/sandbox/api/sift.html).
The main difference is gaussian weighting window parameters.  MP version weights patch center much more (see image below, left) and additionally crops everything outside the circular region. Right is vlfeat version.


![Michal Perdoch kernel](/img/mp_kernel.png)
![vlfeat kernel](/img/vlfeat_kernel.png)



Results:

![hpatches mathing results](/img/hpatches-results.png)


```
OPENCV-SIFT - mAP 
   Easy     Hard      Tough     mean
-------  -------  ---------  -------
0.47788  0.20997  0.0967711  0.26154

VLFeat-SIFT - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.466584  0.203966  0.0935743  0.254708

PYTORCH-SIFT-VLFEAT-65 - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.472563  0.202458  0.0910371  0.255353

NUMPY-SIFT-VLFEAT-65 - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.449431  0.197918  0.0905395  0.245963

PYTORCH-SIFT-MP-65 - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.430887  0.184834  0.0832707  0.232997

NUMPY-SIFT-MP-65 - mAP 
    Easy     Hard      Tough      mean
--------  -------  ---------  --------
0.417296  0.18114  0.0820582  0.226832


```
    
Speed: 
- 0.00246 s per 65x65 patch - [numpy SIFT](https://github.com/ducha-aiki/numpy-sift)
- 0.00028 s per 65x65 patch - [C++ SIFT](https://github.com/perdoch/hesaff/blob/master/siftdesc.cpp)
- 0.00074 s per 65x65 patch - [pytorch SIFT](https://github.com/ducha-aiki/pytorch-sift )CPU, 256 patches per batch
- 0.00038 s per 65x65 patch - [pytorch SIFT](https://github.com/ducha-aiki/pytorch-sift ) GPU (GM940, mobile), 256 patches per batch



If you use this code for academic purposes, please cite the following paper:

    @article{HardNet2017,
     author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
        title = "{Working hard to know your neighbor's margins: Local descriptor learning loss}",
        booktitle = {Proceedings of NIPS},
         year = 2017,
        month = dec}

