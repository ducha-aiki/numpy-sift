import numpy as np
import math
import time

class SIFTDescriptor(object):
    """Class for computing SIFT descriptor of the square patch

    Attributes:
        patchSize: size of the patch in pixels 
        maxBinValue: maximum descriptor element after L2 normalization. All above are clipped to this value
        numOrientationBins: number of orientation bins for histogram
        numSpatialBins: number of spatial bins. The final descriptor size is numSpatialBins x numSpatialBins x numOrientationBins
    """
    def precomputebins(self):
        halfSize = int(self.patchSize/2)
        ps = self.patchSize
        sb = self.spatialBins;
        step = float(self.spatialBins + 1) / (2 * halfSize)
        precomp_bins = np.zeros(2*ps, dtype = np.int32)
        precomp_weights = np.zeros(2*ps, dtype = np.float)
        precomp_bin_weights_by_bx_py_px_mapping = np.zeros((sb,sb,ps,ps), dtype = np.float)
        for i in range(ps):
            i1 = i + ps
            x = step * i
            xi = int(x)
            # bin indices
            precomp_bins[i] = xi -1;
            precomp_bins[i1] = xi
            #bin weights
            precomp_weights[i1] = x - xi;
            precomp_weights[i] = 1.0 - precomp_weights[i1];
            #truncate 
            if  (precomp_bins[i] < 0):
                precomp_bins[i] = 0;
                precomp_weights[i] = 0
            if  (precomp_bins[i] >= self.spatialBins):
                precomp_bins[i] = self.spatialBins - 1;
                precomp_weights[i] = 0
            if  (precomp_bins[i1] < 0):
                precomp_bins[i1] = 0;
                precomp_weights[i1] = 0
            if  (precomp_bins[i1] >= self.spatialBins):
                precomp_bins[i1] = self.spatialBins - 1;
                precomp_weights[i1] = 0
        for y in range(ps):
            for x in range(ps):
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y], precomp_bins[x], y, x ] += precomp_weights[y]*precomp_weights[x]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y+ps], precomp_bins[x], y, x ] += precomp_weights[y+ps]*precomp_weights[x]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y], precomp_bins[x+ps], y, x ] += precomp_weights[y]*precomp_weights[x+ps]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y+ps], precomp_bins[x+ps], y, x ] += precomp_weights[y+ps]*precomp_weights[x+ps]
        if self.mask_type == 'CircularGauss':
            mask = self.CircularGaussKernel(kernlen=self.patchSize, circ=True, sigma_type=self.sigma_type).astype(np.float32)
        elif self.mask_type == 'Gauss':
            mask = self.CircularGaussKernel(kernlen=self.patchSize, circ=False, sigma_type=self.sigma_type).astype(np.float32)
        else:
            raise ValueError(masktype, 'is unknown mask type')

        for y in range(sb):
            for x in range(sb):
                precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:] *= mask
                precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:] = np.maximum(0,precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:])
        return precomp_bins.astype(np.int32),precomp_weights,precomp_bin_weights_by_bx_py_px_mapping,mask
    def __repr__(self):
            return self.__class__.__name__ + '(' + 'orientationBins=' + str(self.orientationBins) +\
             ', ' + 'spatialBins=' + str(self.spatialBins) +\
             ', ' + 'patchSize=' + str(self.patchSize) +\
             ', ' + 'sigma_type=' + str(self.sigma_type) +\
             ', ' + 'mask_type=' + str(self.mask_type) +\
             ', ' + 'maxBinValue=' + str(self.maxBinValue) + ')'

    def __init__(self, patchSize = 41,
                 maxBinValue = 0.2,
                 numOrientationBins = 8,
                 numSpatialBins = 4,
                 mask_type = 'CircularGauss',
                 sigma_type = 'hesamp'):
        self.patchSize = patchSize
        self.maxBinValue = maxBinValue
        self.orientationBins = numOrientationBins
        self.spatialBins = numSpatialBins
        self.mask_type = mask_type
        self.sigma_type = sigma_type
        self.precomp_bins,self.precomp_weights,self.mapping,self.mask = self.precomputebins()
        self.binaryMask = self.mask > 0
        self.gx = np.zeros((patchSize,patchSize), dtype=np.float)
        self.gy = np.zeros((patchSize,patchSize), dtype=np.float)
        self.ori = np.zeros((patchSize,patchSize), dtype=np.float)
        self.mag = np.zeros((patchSize,patchSize), dtype=np.float)
        self.norm_patch = np.zeros((patchSize,patchSize), dtype=np.float)
        ps = self.patchSize
        sb = self.spatialBins
        ob = self.orientationBins
        self.desc = np.zeros((ob, sb , sb ), dtype = np.float)
        return
    def CircularGaussKernel(self,kernlen=21, circ = True, sigma_type = 'hesamp'):
        halfSize = float(kernlen) / 2.;
        r2 = float(halfSize**2);
        if sigma_type == 'hesamp':
            sigma_mul_2 = 0.9 * r2;
        elif sigma_type == 'vlfeat':
            sigma_mul_2 = kernlen**2
        else:
            raise ValueError('Unknown sigma_type', sigma_type, 'try hesamp or vlfeat')
        disq = 0;
        kernel = np.zeros((kernlen,kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize+0.5)**2 +  (x - halfSize+0.5)**2;
                kernel[y,x] = math.exp(-disq / sigma_mul_2)
                if circ and (disq >= r2):
                    kernel[y,x] = 0.
        return kernel
    def photonorm(self, patch, binaryMask = None):
        if binaryMask is not None:
            std1_coef = 50. /  np.std(patch[binaryMask])
            mean1 =  np.mean(patch[binaryMask])
        else:
            std1_coef = 50. / np.std(patch)
            mean1 =  np.mean(patch)
        if std1_coef >= 50. / 0.000001:
            std1_coef = 50.0
        self.norm_patch = 128. + std1_coef * (patch - mean1);
        self.norm_patch = np.clip(self.norm_patch, 0.,255.);
        return
    def getDerivatives(self,image):
        #[-1 1] kernel for borders
        self.gx[:,0] = image[:,1] - image[:,0]
        self.gy[0,:] = image[1,:] - image[0,:]
        self.gx[:,-1] = image[:,-1] - image[:,-2]
        self.gy[-1,:] = image[-1,:] - image[-2,:]
        #[-1 0 1] kernel for the rest
        self.gy[1:-2,:] = image[2:-1,:] - image[0:-3,:]
        self.gx[:,1:-2] = image[:,2:-1] - image[:,0:-3]
        self.gx *= 0.5
        self.gy *= 0.5
        return
    def samplePatch(self,grad,ori):
        ps = self.patchSize
        sb = self.spatialBins
        ob = self.orientationBins
        o_big = float(ob) * (ori + 2.0*math.pi) / (2.0 * math.pi)
        bo0_big = np.floor(o_big)#.astype(np.int32)
        wo1_big = o_big - bo0_big;
        bo0_big = bo0_big % ob;
        bo1_big = (bo0_big + 1.0) % ob;
        wo0_big = 1.0 - wo1_big;
        wo0_big *= grad;
        wo0_big = np.maximum(0, wo0_big)
        wo1_big *= grad;
        wo1_big = np.maximum(0, wo1_big)
        ori_weight_map = np.zeros((ob,ps,ps))
        for o in range(ob):
            relevant0 = np.where(bo0_big == o)
            ori_weight_map[o, relevant0[0], relevant0[1]] = wo0_big[relevant0[0], relevant0[1]]
            relevant1 = np.where(bo1_big == o)
            ori_weight_map[o, relevant1[0], relevant1[1]] += wo1_big[relevant1[0], relevant1[1]]
        for y in range(sb):
            for x in range(sb):
                self.desc[:,y,x] =  np.tensordot( ori_weight_map, self.mapping[y,x,:,:])
        return
    def describe(self,patch, userootsift = False, flatten = True, show_timings = False):
        t = time.time()
        self.photonorm(patch, binaryMask = self.binaryMask);
        if show_timings:
            print( 'photonorm time = ', time.time() - t)
            t = time.time()
        self.getDerivatives(self.norm_patch)
        if show_timings:
            print( 'gradients time = ', time.time() - t)
            t = time.time()
        self.mag = np.sqrt(self.gx * self.gx + self.gy*self.gy)
        self.ori = np.arctan2(self.gy,self.gx)
        if show_timings:
            print( 'mag + ori time = ', time.time() - t)
            t = time.time()
        self.samplePatch(self.mag,self.ori)
        if show_timings:
            print( 'sample patch time = ', time.time() - t)
            t = time.time()
        self.desc /= np.linalg.norm(self.desc.flatten(),2)
        self.desc = np.clip(self.desc, 0,self.maxBinValue);
        self.desc /= np.linalg.norm(self.desc.flatten(),2)
        if userootsift:
            self.desc = np.sqrt(self.desc / np.linalg.norm(unnorm_desc.flatten(),1))
        if show_timings:
            print( 'clip and norm time = ', time.time() - t)
            t = time.time()
        if flatten:
            return np.clip(512. * self.desc.flatten() , 0, 255).astype(np.int32);
        else:
            return np.clip(512. * self.desc , 0, 255).astype(np.int32);
