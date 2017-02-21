import numpy as np
import math

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
        mask =  self.CircularGaussKernel(kernlen=self.patchSize)
        for y in range(sb):
            for x in range(sb):
                precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:] *= mask
                precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:] = np.maximum(0,precomp_bin_weights_by_bx_py_px_mapping[y,x,:,:])
        return precomp_bins.astype(np.int32),precomp_weights,precomp_bin_weights_by_bx_py_px_mapping,mask
    def __init__(self, patchSize = 41, maxBinValue = 0.2, numOrientationBins = 8, numSpatialBins = 4):
        self.patchSize = patchSize
        self.maxBinValue = maxBinValue
        self.orientationBins = numOrientationBins
        self.spatialBins = numSpatialBins
        self.precomp_bins,self.precomp_weights,self.mapping,self.mask = self.precomputebins()
        self.binaryMask = self.mask > 0
    def CircularGaussKernel(self, kernlen=21):
        halfSize = kernlen / 2;
        r2 = halfSize*halfSize;
        sigma2 = 0.9 * r2;
        disq = 0;
        kernel = np.zeros((kernlen,kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize)*(y - halfSize) +  (x - halfSize)*(x - halfSize);
                if disq < r2:
                    kernel[y,x] = math.exp(-disq / sigma2)
                else:
                    kernel[y,x] = 0
        return kernel

    def photonorm(self, patch, binaryMask = None):
        if binaryMask is not None:
            std1 = np.std(patch[binaryMask])
            mean1 =  np.mean(patch[binaryMask])
        else:
            std1 = np.std(patch)
            mean1 =  np.mean(patch)
        if std1 <= 0.000001:
            std1 = 1.0 
        outpatch = 128. + 50.*(patch - mean1) / std1;
        outpatch = np.clip(outpatch, 0.,255.);
        return outpatch
    def getDerivatives(self,image):
        sh = image.shape
        gx = np.zeros(sh, dtype=np.float)
        gy = np.zeros(sh, dtype=np.float)
        for y in range(sh[0]):
            if y == 0:
                gx[:,y] = image[:,1] - image[:,0]
                gy[y,:] = image[1,:] - image[0,:]
            elif y == sh[0] - 1:
                gx[:,y] = image[:,-1] - image[:,-2]
                gy[y,:] = image[-1,:] - image[-2,:]
            else:
                gy[y,:] = image[y+1,:] - image[y-1,:]
                gx[:,y] = image[:,y+1] - image[:,y-1]    
        return 0.5 * gx, 0.5 * gy
    def samplePatch(self,grad,ori):
        ps = self.patchSize
        sb = self.spatialBins
        ob = self.orientationBins
        desc = np.zeros((ob, sb , sb ), dtype = np.float)
        o_big = float(ob) * (ori + 2.0*math.pi) / (2.0 * math.pi)
        bo0_big = np.floor(o_big)#.astype(np.int32)
        wo1_big = o_big - bo0_big;
        bo0_big = bo0_big % ob;
        bo1_big = (bo0_big + 1.0) % ob;
        wo0_big = 1.0 - wo1_big;
        wo0_big *= grad;
        wo1_big *= grad;
        ori_weight_map = np.zeros((ob,ps,ps))
        for o in range(ob):
            relevant0 = bo0_big == o
            ori_weight_map[o,relevant0] = np.maximum(0,wo0_big[relevant0])
            relevant1 = bo1_big == o
            ori_weight_map[o,relevant1] += np.maximum(wo1_big[relevant1],0)
        for y in range(sb):
            for x in range(sb):
                current_val = self.mapping[y,x,:,:]
                for o in range(ob):
                    desc[o,y,x] = np.sum(current_val * ori_weight_map[o,:,:])
        return desc
    def describe(self,patch, userootsift = False):
        norm_patch = self.photonorm(patch, binaryMask = self.binaryMask);
        gx,gy = self.getDerivatives(norm_patch)
        mag = np.sqrt(gx * gx + gy*gy)
        ori = np.arctan2(gy,gx)
        unnorm_desc = self.samplePatch(mag,ori)
        unnorm_desc /= np.linalg.norm(unnorm_desc[:],2)
        unnorm_desc = np.clip(unnorm_desc, 0,self.maxBinValue);
        unnorm_desc /= np.linalg.norm(unnorm_desc[:],2)
        if userootsift:
            unnorm_desc = np.sqrt(unnorm_desc / np.linalg.norm(unnorm_desc.flatten(),1))
        return np.clip(512. * unnorm_desc , 0, 255).astype(np.int32);