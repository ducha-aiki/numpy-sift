import sys
import argparse
import time
import os
import sys
import cv2
import math
import numpy as np
from tqdm import tqdm
from numpy_sift import SIFTDescriptor
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os

assert len(sys.argv)==3, "Usage python hpatches_extract_numpysift.py hpatches_db_root_folder 64"
OUT_W = int(sys.argv[2])    
# all types of patches 
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))
            
    
seqs = glob.glob(sys.argv[1]+'/*')
seqs = [os.path.abspath(p) for p in seqs]     

descr_name = 'numpy-sift-'+str(OUT_W)

model = SIFTDescriptor(patchSize = OUT_W)


for seq_path in seqs:
    seq = hpatches_sequence(seq_path)
    path = os.path.join(descr_name,seq.name)
    if not os.path.exists(path):
        os.makedirs(path)
    descr = np.zeros((int(seq.N),128)) # trivial (mi,sigma) descriptor
    for tp in tps:
        print(seq.name+'/'+tp)
        if os.path.isfile(os.path.join(path,tp+'.csv')):
            continue
        n_patches = 0
        for i,patch in enumerate(getattr(seq, tp)):
            n_patches+=1
        t = time.time()
        descriptors = np.zeros((n_patches, 128))
        if OUT_W != 65:
            for i,patch in enumerate(getattr(seq, tp)):
                descriptors[i,:] = model.describe(cv2.resize(patch,(OUT_W,OUT_W)))
        else:
            for i,patch in enumerate(getattr(seq, tp)):
                descriptors[i,:] = model.describe(patch)
        np.savetxt(os.path.join(path,tp+'.csv'), descriptors.astype(np.uint8), delimiter=',', fmt='%d')
