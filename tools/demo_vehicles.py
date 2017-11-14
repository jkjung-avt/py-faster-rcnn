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
from fast_rcnn.nms_wrapper import soft_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

CLASSES = ('__background__', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck')
COLORS = ((0,0,0), (0,76,153), (0,0,255), (51,153,255), (0,255,255), (0,204,0), (204,0,0))

def vis_detections(im, class_name, dets, thresh=0.5):
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
        #ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_detections_cv(im_name, im, dets_all, cls_all, thresh=0.5):
    """Draw detected bounding boxes."""
    assert len(dets_all) == len(cls_all)
    windowName = 'demo_vehicles'
    inds = np.where(dets_all[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets_all[i, :4]
        score = dets_all[i, -1]
        cls = cls_all[i]
        cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), COLORS[cls], 2)
        txt = '{:s} {:.3f}'.format(CLASSES[cls], score)
        cv2.putText(im, txt, (int(bbox[0])+1,int(bbox[1])-2), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(im, txt, (int(bbox[0]),int(bbox[1])-2), cv2.FONT_HERSHEY_PLAIN, 1.0, (240,240,240), 1, cv2.LINE_AA)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    height, width, colors = im.shape
    cv2.resizeWindow(windowName, width, height)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, im_name)
    cv2.imshow(windowName, im)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo/vehicles', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    det_time = timer.toc(average=False)

    #print('im_detect(): output {} boxes'.format(boxes.shape[0]))
    timer.tic()
    dets_list = []
    cls_list = []
    CONF_THRESH = 0.7
    NMS_THRESH = 0.2
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = [i for i in range(dets.shape[0])]
        keep = soft_nms(dets=dets, Nt=NMS_THRESH, method=1)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds]
        dets_list.append(dets)
        cls_list.extend([cls_ind] * len(dets))
    dets_all = np.concatenate(dets_list, axis=0)
    nms_time = timer.toc(average=False)
    print('Detection took {:.3f}s and found {:d} objects'.format(timer.total_time, len(dets_all)))
    vis_detections_cv(im_name, im, dets_all, cls_list, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=['zf','vgg16','googlenet'], default='vgg16')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.MAX_SIZE = 1920
    cfg.TEST.SCALES = (1080,)
    cfg.TEST.RPN_POST_NMS_TOP_N = 1000

    args = parse_args()

    if args.demo_net == 'vgg16':
        prototxt = 'models/vehicles/VGG16/faster_rcnn_end2end/test.prototxt'
        caffemodel = 'data/faster_rcnn_models/vehicles_vgg16_iter_490000.caffemodel'
    elif args.demo_net == 'googlenet':
        prototxt = 'models/vehicles/GoogLeNet/faster_rcnn_end2end/test.prototxt'
        caffemodel = 'data/faster_rcnn_models/vehicles_googlenet_iter_490000.caffemodel'
    else:
        sys.exit('A valid network model has not been specified!')

    if not os.path.isfile(caffemodel):
        raise IOError('{:s} not found.'.format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((640, 480, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    #im_names = ['10.png', '15.png', '20.png', '25.png']
    im_dir = os.path.join(cfg.DATA_DIR, 'demo/vehicles')
    im_names = [f for f in os.listdir(im_dir) if f.endswith('.jpg')]
    #import random
    #im_names = random.sample(im_names, 10)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
