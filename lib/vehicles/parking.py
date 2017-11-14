#
#
#

import numpy as np
import os, sys, datetime
import _init_paths
from fast_rcnn.config import cfg
#from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import soft_nms
import cv2
import caffe

# If 2 detections in neighboring images overlap over this amount, they
# are considered as 'same detection in different image frames'.
OVERLAP_THRESHOLD = 0.8

# If 2 image crops have similarity scores over this value, they are
# considered as 'sam object in different image crops'. This threshold
# is needed due to occlusion, fast lighting condition changes, rain, etc.
SIMILARITY_THRESHOLD = 0.7

# If a vehicle is present over this percentage of recent image frames,
# it is considered as a stationary vehicle.
# (stationary -> alarming -> violation)
PRESENCE_THRESHOLD = 0.7

CLASSES = ('__background__', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck')
CLASSES_FOR_DETECTION = ('motorcycle', 'bus', 'truck')

def im_detect(net, img):
    """
    For testing only.
    """
    boxes = np.zeros((2, 4*7), dtype=np.float32)
    boxes[0, 8:12] = np.array([100,100,300,300])
    boxes[1, 8:12] = np.array([800,800,1200,1000])
    scores = np.zeros((2, 7), dtype=np.float32)
    scores[0, 2] = 0.99
    scores[0, 0] = 0.01
    scores[1, 2] = 0.99
    scores[1, 0] = 0.01
    return scores, boxes

def calculate_iou(det1, det2):
    """
    'iou': Intersection Over Union
           iou=1 -> complete overlap
           iou=0 -> no overlap at all
    """
    x_overlap = max(0, min(det1[2], det2[2]) - max(det1[0], det2[0]))
    y_overlap = max(0, min(det1[3], det2[3]) - max(det1[1], det2[1]))
    if x_overlap <= 0 or y_overlap <= 0:
        return 0
    overlap_area = x_overlap * y_overlap
    area1 = (det1[2]-det1[0])*(det1[3]-det1[1])
    area2 = (det2[2]-det2[0])*(det2[3]-det2[1])
    unionarea = area1 + area2 - overlap_area
    return overlap_area / float(unionarea)

def calculate_similarity(crop1, crop2):
    """
    Calculate similarity between 2 image crops using image features.
    Assumes the 2 crops are of the same dimension/size and in (H, W, C)
    format.
    """
    return 1.0

class VehicleDetected(object):
    def __init__(self, timestamp, det):
        slef.timestamp = timestamp
        self.det = det  # detection box (x/y coordinates and score)

class IllegalParkingDetector(object):
    """
    'img_list'       holds up to 'ALARMING_PERIOD' number of most recent
                     images.
    'alarming_list'  holds vehicles which have been at the same place
                     for over ALARMING_INTERVAL but not yet as long as
                     VIOLATION_INERVAL.
    'violation_list' holds vehicles which have been staying at the same
                     location for longer than VIOLATION_INTERVAL.
    """
    def __init__(self, net):
        self.DETECTION_INTERVAL = datetime.timedelta(0, 10, 0)  # 10 seconds
        self.ALARMING_INTERVAL  = datetime.timedelta(1,  0, 0)  # 1 minute
        self.ALARMING_PERIOD    = 6 # ALARMING_INTERVAL = 6 * DETECTION_INTERVAL
        self.VIOLATION_INTERVAL = datetime.timedelta(5,  0, 0)  # 5 minutes
        self.PRESENCE_FRAMES    = int(self.ALARMING_PERIOD * PRESENCE_THRESHOLD)
        self.img_list = []        # list of (img, timestamp, dte) tuples
        self.alarming_list = []   # list of 'VehicleDetected' objects
        self.violation_list = []  # list of 'VehicleDetected' objects

        self.net = net  # the Caffe network (Faster RCNN)

    def detect(self, net, img):
        """Detect objects in an image."""
        scores, boxes = im_detect(self.net, im)
        dets_list = []
        CONF_THRESH = 0.7
        NMS_THRESH = 0.2
        for cls_ind, cls in enumerate(CLASSES):
            if cls in CLASSES_FOR_DETECTION:
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = soft_nms(dets=dets, Nt=NMS_THRESH, method=1)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                dets_list.append(dets[inds])
        dets_all = np.concatenate(dets_list, axis=0)
        print('Detection found {:d} objects'.format(len(dets_all)))
        return dets_all

    def new_image(self, img):
        """
        Process a new image. Every time the main thread gets a new image
        from camera, it calls this method. This function checks whether
        the new image frame has arrived around DETECTION_INTERVAL since
        the previous recorded image frame, ad then processes the new
        image frame accordingly.
        """
        curr_time = datetime.datetime.now()
        if len(img_list) > 0:
            prev_time = img_list[-1][1]
            if prev_time - curr_time < self.DETECTION_INTERVAL:
                return  # has not reached the next detection period
            assert(prev_time - curr_time < self.DETECTION_INTERVAL * 2)
        dets = self.detect(img)
        self.img_list.append((img, curr_time, dets))
        if len(self.img_list) < self.ALARMING_PERIOD:
            return  # need to accumulate a few more images
        elif len(self.img_list) > self.ALARMING_PERIOD:
            self.img_list.remove(self.img_list[0])
            assert(len(self.img_list) == self.ALARMING_PERIOD)

        self.update_vehicleslist()

    def det_already_in_list(self, det, new_list):
        """
        Check whether a 'det' (detection) is in 'new_list' already.
        The criteria is whether 'det' overlaps with another existing
        box over the threshold.
        """
        found = False
        index = 0
        for i, existing in enumerate(new_list):
            if calculate_iou(det, exsiting) >= OVERLAP_THRESHOLD:
                found = True
                index = i
                break
        return found, index

    def vehicle_in_list(self, vehicle, det_list):
        return self.det_already_in_list(vehicle.det, det_list)

    def match_crops(self, crop_list):
        """
        Check if crops in the list match one another with a ratio
        above PRESENCE_THRESHOLD.
        """
        assert len(crop_list) == self.ALARMING_PERIOD
        for c1 in crop_list:
            mscores = []
            count = 0
            for c2 in crop_list:
                if c1 != c2:
                    mscores.append(calculate_similarity(c1, c2))
                    if mscores[-1] >= SIMILARITY_THRESHOLD:
                        count += 1
            avg = sum(mscores) / float(len(mscores))
            if avg >= SIMILARITY_THRESHOLD and count >= self.PRESENCE_FRAMES:
                return True
        return False

    def update_vehicleslist(self):
        # Step 1: Check detection boxes in all stored image frames. Keep
        #         detection boxes which appear at the same location in
        #         different image frames multiple times.
        new_list = []
        cnt_list = []
        for i, v in enumerate(reversed(self.img_list)):
            dets = v[-1]
            for j in range(len(dets)):
                found, index = self.det_already_in_list(dets[j], new_list)
                if found:
                    cnt_list[index] += 1
                else
                    new_list.append(det)
                    cnt_list.append(1)
        # Only keep boxes which appear more than PRESENCE_FRAMES times
        keep = np.where(np.array(cnt_list) >= self.PRESENCE_FRAMES)[0]
        det_list = []
        for i in range(len(keep)):
            det_list.append(new_list[keep[i]])
        print('update_vehicleslist() found {} presences'.format(len(det_list)))

        # Step 2: Further check whether image crops at the detection
        #         locations match to certain extent.
        tmp_list = []
        for det in det_list:
            crop_list = [x[0][det[1]:det[3],det[0]:det[2],:] for x in self.img_list]
            if self.match_crops(crop_list): 
                tmp_list.append(det)
        det_list = tmp_list

        # Step 3: Check and update violation list.
        tmp_list = []
        for v in self.violation_list:
            found, index = self.vehicle_in_list(v, det_list):
            if found:
                det_list.remove(det_list(index))
                tmp_list.append(v)
            else:
                print('update_vehicleslist(): ({},{},{},{}) removed from violation list.'.format(v.det[0],v.det[1],v.det[2],v.det[3])
        self.violation_list = tmp_list

        # Step 4: Update alarming list.
        tmp_list = []
        for v in self.alarming_list:
            found, index = self.vehicle_in_list(v, det_list):
            if found:
                det_list.remove(det_list(index))
                curr_time = datetime.datetime.now()
                if curr_time - v.timestamp >= self.VIOLATION_INTERVAL:
                    self.violation_list.append(v)
                    print('update_vehicleslist(): ({},{},{},{}) moved from alarming to violation list.'.format(v.det[0],v.det[1],v.det[2],v.det[3])
                else:
                    tmp_list.append(v)
            else:
                print('update_vehicleslist(): ({},{},{},{}) removed from alarming list.'.format(v.det[0],v.det[1],v.det[2],v.det[3])
        self.alarming_list = tmp_list

        # Step 5: Out all remaining detections into alarming list.
        for det in det_list:
            timestamp = datetime.datetime.now() - self.ALARMING_INERVAL
            v = VehicleDetected(timestamp, det)
            self.alarming_list.append(v)
            print('update_vehicleslist(): ({},{},{},{}) appended into alarming list.'.format(v.det[0],v.det[1],v.det[2],v.det[3])

# Testing code
if __name__ == '__main__':
    windowName = 'vehicles_test'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1920, 1080)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, "Testing IllegalParkingDetector")

    cnt = 1
    ipd = IllegalParkingDetector(None)
    while True:
        filename = './lib/vehicels/test/img{:03d}.png'.format(cnt)
        img = cv2.imread(filename)
        ipd.new_image(img)

        ### show detection results
        print('******')
        print('alarming list:')
        if len(ipd.alarming_list) == 0:
            print('  None')
        else
            for i, v in enumerate(ipd.alarming_list):
                print('  #{}: ({},{},{},{})'.format(i, v.det[0], v.det[1], v.det[0], v.det[3]))
        print('violation list:')
        if len(ipd.alarming_list) == 0:
            print('  None')
        else
            for i, v in enumerate(ipd.violation_list):
                print('  #{}: ({},{},{},{})'.format(i, v.det[0], v.det[1], v.det[0], v.det[3]))

        key = cv2.waitKey(1000)  # 1 second
        if key == 27: # ESC key: quit program
            break
        cnt += 1
        if cnt > 12:
            cne = 1

    cv2.destroyAllWindows()
