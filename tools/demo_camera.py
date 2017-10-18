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
#import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

CLASSES = ('__background__', 'Head')
windowName = "HeadDetection"

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        #ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')
    return len(inds)

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:2]): # show result of class 1 only
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    found = vis_detections(im, cls, dets, thresh=CONF_THRESH)
    print ('Detection took {:.3f}s and found {:d} objects').format(timer.total_time, found)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Brainwash Faster R-CNN demo with live camera feed')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=['zf','vgg16','googlenet'], default='vgg16')
    parser.add_argument("--rtsp", dest="use_rtsp",
                        help="use IP CAM (remember to also set --uri)",
                        action="store_true")
    parser.add_argument("--uri", dest="rtsp_uri",
                        help="RTSP URI string, e.g. rtsp://192.168.1.64:554",
                        default=None, type=str)
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam (remember to also set --vid)",
                        action="store_true")
    parser.add_argument("--vid", dest="video_dev",
                        help="video device # of USB webcam (/dev/video?) [1]",
                        default=1, type=int)
    parser.add_argument("--width", dest="image_width",
                        help="image width [640]",
                        default=640, type=int)
    parser.add_argument("--height", dest="image_height",
                        help="image width [480]",
                        default=480, type=int)
    args = parser.parse_args()
    return args

def open_cam_rtsp(uri, width, height):
    gst_str = "rtspsrc location={} latency=50 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink".format(uri, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = "v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! videoconvert ! appsink".format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_onboard(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink".format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_window(width, height):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, width, height)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, "'Head' Detection Demo")

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.demo_net == 'zf':
        prototxt = 'models/brainwash/ZF/faster_rcnn_end2end/test.prototxt'
        caffemodel = 'data/faster_rcnn_models/brainwash_zf_iter_70000.caffemodel'
    elif args.demo_net == 'vgg16':
        prototxt = 'models/brainwash/VGG16/faster_rcnn_end2end/test.prototxt'
        caffemodel = 'data/faster_rcnn_models/brainwash_vgg16_iter_70000.caffemodel'
    elif args.demo_net == 'googlenet':
        prototxt = 'models/brainwash/GoogLeNet/faster_rcnn_end2end/test.prototxt'
        caffemodel = 'data/faster_rcnn_models/brainwash_googlenet_iter_90000.caffemodel'
    else:
        sys.exit('A valid network model has not been specified!')

    if not os.path.isfile(caffemodel):
        raise IOError('{:s} not found.')

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

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri, args.image_width, args.image_height)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev, args.image_width, args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width, args.image_height)

    if not cap.isOpened():
        sys.exit("Failed to open camera!")

    open_window(args.image_width, args.image_height)


    showHelp = True
    font = cv2.FONT_HERSHEY_PLAIN
    helpText="'Esc' to Quit, 'H' to Toggle Help, 'F' to Toggle Fullscreen"
    showFullScreen = False
    while True:
        if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
            # This will fail if the user closed the window; Nasties get printed to the console
            break;
        ret_val, displayBuf = cap.read();
        demo(net, displayBuf) # bounding boxes would be drawn on 'displayBuf' directly
        if showHelp == True:
            cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
            cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
        cv2.imshow(windowName, displayBuf)
        key = cv2.waitKey(10)
        if key == 27: # Check for ESC key
            cv2.destroyAllWindows()
            break
        elif key == 72 or key == 104: # 'H'/'h': toggle help message
            showHelp = not showHelp
        elif key == 70 or key == 102: # 'F'/'f': toggle fullscreen
            if showFullScreen == False : 
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) 
            showFullScreen = not showFullScreen

