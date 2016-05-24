#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""
from PIL import Image
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',)
synsets = sio.loadmat(os.path.join('/home/breathe/Desktop/ImageNet/', 'data', 'meta_det.mat'))
for i in xrange(200):
    CLASSES = CLASSES + (synsets['synsets'][0][i][2][0],)

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_100000_from_AndrewLiao.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':


    args = parse_args()


    dir =   '/home/breathe/Desktop/ImageNet/ILSVRC2014_DET_train'
  
    for directories in os.listdir(dir):
        if directories.startswith("n") and not directories.endswith('.tar'):
            dir_path = os.path.join(dir, directories) 
            for file in os.listdir(dir_path):
                if file.endswith(".JPEG"):
		    
                    im_file = os.path.join(dir_path, file)

		    print(' file {}'.format(im_file))
		    
		    
                    file = open(im_file, 'rb')
                    b = bytearray(file.read())
                    size  = len(b)
                    if (b[0] != 0xFF) or (b[1] != 0xD8) or (b[size - 2] != 0xFF) or (b[size - 1] != 0xD9):
			
                        print('wrong format with file {}'.format(im_file))
                        print('{} {} {} {}'.format(b[0],b[1],b[size - 2],b[size - 1]))
			continue
                    img = Image.open(im_file)
                    iw,ih = img.size
                    aspect = iw / float (ih)
		   
                    if (aspect < 0.117) or (aspect > 15.5):
			
                        print('small aspect with file {}'.format(im_file))
			continue
		    smallest = 20
		    if iw<smallest or ih<smallest:	
					
			print('too small with file {} {} {}'.format(iw, ih,im_file))
			continue
                    im = cv2.imread(im_file)
		    im = im.astype(np.float32, copy=False)
		    del img

    print i
