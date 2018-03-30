#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# --------------------------------------------------------
# Demo script showing real-time inferencing result with
# live camera input.
# Modified from the original demo.py by JK Jung
# --------------------------------------------------------


import os
import sys
import argparse
import numpy as np
import cv2
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe, os, sys, cv2


VOC_CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
COLORS = ((0,0,0),
          (0,0,204), (0,102,204), (0,204,204), (0,204,0),
          (204,204,0), (204,102,0), (204,0,0), (204,0,102),
          (204,0,102), (102,0,204), (153,153,255), (153,204,255),
          (153,255,255), (153,255,153), (255,255,153), (255,204,153),
          (255,153,153), (255,153,204), (255,153,255), (204,153,255))
DEFAULT_PROTOTXT = 'models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
DEFAULT_MODEL    = 'data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
windowName = 'FRCN-demo'


def vis_detections_cv(im, det_list, cls_list, thresh=0.5):
    """Draw detected bounding boxes."""
    assert len(det_list) == len(cls_list)
    for i in range(len(det_list)):
        bbox = det_list[i][:4]
        score = det_list[i][-1]
        cls = cls_list[i]
        cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), COLORS[cls], 2)
        txt = '{:s} {:.2f}'.format(VOC_CLASSES[cls], score)
        cv2.putText(im, txt, (int(bbox[0])+1,int(bbox[1])-2), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32,32), 4, cv2.LINE_AA)
        cv2.putText(im, txt, (int(bbox[0]),int(bbox[1])-2), cv2.FONT_HERSHEY_PLAIN, 1.0, (240,240,240), 1, cv2.LINE_AA)
    return im


def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()

    det_list = []  # list of final detection boxes and scores
    cls_list = []  # list of detection class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(VOC_CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            det_list.append(dets[i])
        cls_list.extend([cls_ind] * len(inds))
    print('Detection took {:.3f}s and found {:d} objects').format(timer.total_time, len(det_list))
    return vis_detections_cv(im, det_list, cls_list, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster RCNN demo with camera')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI string, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='video device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [640]',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image width [480]',
                        default=480, type=int)
    parser.add_argument('--prototxt', dest='caffe_prototxt',
                        help='caffe prototxt of the Faster RCNN model [{}]'.format(DEFAULT_PROTOTXT),
                        default=DEFAULT_PROTOTXT, type=str)
    parser.add_argument('--model', dest='caffe_model',
                        help='the caffemodel (weights) file [{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_MODEL, type=str)
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! videoconvert ! appsink'.format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink'.format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(windowName, width, height)
    #cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, 'Faster RCNN camera demo')


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if not os.path.isfile(args.caffe_prototxt):
        sys.exit('{} not found!'.format(args.caffe_prototxt))
    if not os.path.isfile(args.caffe_model):
        sys.exit('{} not found!'.format(args.caffe_model))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(args.caffe_prototxt, args.caffe_model, caffe.TEST)

    # Warm-up with 2 dummy images
    im = 128 * np.ones((640, 480, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri, args.image_width, args.image_height, args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev, args.image_width, args.image_height)
    else:  # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width, args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)
    while True:
        if cv2.getWindowProperty(windowName, 0) < 0: break  # display window terminated by user
        ret_val, img = cap.read();
        if not ret_val: break  # image capture failed
        img = demo(net, img)
        cv2.imshow(windowName, img)
        key = cv2.waitKey(1)
        if key == 27: break  # ESC key pressed

    cap.release()
    cv2.destroyAllWindows()
