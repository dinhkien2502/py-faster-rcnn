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

def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format('hand', score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image
    fig = plt.gcf()
    fig.savefig("output_"+image_name)


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo','hands','GTEA_S2', image_name)



    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.92
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)
	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    	if len(inds) == 0:
        	continue
    	for i in inds:
        	bbox = dets[i, :4]
        	score = dets[i, -1]

        	ax.add_patch(
            		plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            	)
        	ax.text(bbox[0], bbox[1] - 2,
                	'{:s} {:.3f}'.format(cls, score),
                	bbox=dict(facecolor='blue', alpha=0.5),
                	fontsize=14, color='white')

    	
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image
    #fig = plt.gcf()
    #output_dir = os.path.join(cfg.ROOT_DIR,'data','demo_hands_out','iter90k','GTEA_S2',"output_"+image_name)
    #fig.savefig(output_dir)

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
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt')
    #caffemodel = '/home/andrewliao11/Tracker/vgg16_faster_rcnn_iter_100000.caffemodel'
    
    #caffemodel = os.path.join(cfg.DATA_DIR, 'my_models/vgg16_faster_rcnn_iter_10000_hands.caffemodel')
    caffemodel = '/home/breathe/Desktop/Object_Detection/Faster-RCNN-200DET-Cuc/py-faster-rcnn/output/seed_rng5103/train/vgg16_faster_rcnn_iter_90000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    #print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)



    dir = os.path.join(cfg.ROOT_DIR,'data','demo','hands','GTEA_S2')
    for file in os.listdir(dir):
        if file.endswith(".png"):
            demo(net, file)

    #plt.show()
