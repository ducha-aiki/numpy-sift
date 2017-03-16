This is an python implementation of SIFT patch descriptor. 
It is derived from Michal Perdoch C++ implementation at https://github.com/perdoch/hesaff

The SIFT descriptor code is protected under a US Patent 6,711,293. A
license MUST be obtained from the University of British Columbia for
use of SIFT code, files numpy_sift.py, in commercial
applications (see LICENSE.SIFT for details)

Here are comparisons to the original Michal Perdoch implementation. 
The benchmark is on W1BS dataset from [WxBS: Wide Baseline Stereo Generalizations](https://arxiv.org/abs/1504.06603.pdf) paper, figure 3. So there is no difference between versions in performance 

Average performance on W1BS

![average](/img/total.png)
    
Speed: 
- 0.00187 s per 65x65 patch - numpy SIFT
- 0.00028 s per 65x65 patch - C++ SIFT

If you use this code for academic purposes, please cite the following paper:

    @article {tbd}
