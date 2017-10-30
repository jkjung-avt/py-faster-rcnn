from datasets.imdb import imdb
from fast_rcnn.config import cfg
import os
import errno
import os.path as osp
import json
import uuid
import scipy.sparse
import scipy.io as sio
import numpy as np
import cPickle
from vehicles_eval import vehicles_eval

class vehicles(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'vehicles_' + image_set)
        self._image_set = image_set  # 'train' or 'test'
        self._data_path = osp.join(cfg.DATA_DIR, 'vehicles')
        fname = osp.join(self._data_path, image_set + '.json')
        with open(fname, 'rb') as f:
            self._image_index = json.load(f)
            self._classes = ['__background__', 'bicycle', 'car',
                             'motorcycle', 'bus', 'train', 'truck', 'boat']
            for img in self._image_index:
                for anno in img['annotations']:
                    #if anno['class'] not in self._classes:
                    #    self._classes.append(anno['class'])
                    assert anno['class'] in self._classes, \
                           'Unknown class "{}" in {}'.format(anno['class'], img)
            #print(self.classes)
            print('Total number of classes: {}'.format(self.num_classes))
            self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
            print(self._class_to_ind)
            self._salt = str(uuid.uuid4())
            self._comp_id = 'comp4'
            # Specific config options
            self.config = {'cleanup'  : False,
                           'use_salt' : True,
                           'top_k'    : 2000,
                           'use_diff' : False,
                           'rpn_file' : None}

    def image_path_at(self, i):
        return self._image_index[i]['filename']

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_vehicles_annotation(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        print('### vehicles.rpn_roidb() called!')
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        #roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        print('### vehicles._load_rpn_roidb() called!')
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_vehicles_annotation(self, item):
        num_objs = 0
        for anno in item['annotations']:
            if 'width' in anno and 'height' in anno:
                num_objs = num_objs + 1
        assert num_objs > 0
        if num_objs == 0:
            print('### No objects in {}!'.format(item['filename']))
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ix = 0
        for anno in item['annotations']:
            if 'width' in anno and 'height' in anno:
                x, y, width, height = anno['x'], anno['y'], anno['width'], anno['height']
                cls = self._class_to_ind[anno['class']]
                boxes[ix, :] = [x, y, x + width, y + height]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = width * height
                ix = ix + 1
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_vehicles_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._data_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._data_path,
            'results',
            filename)
        return path

    def _write_vehicles_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} vehicles results file'.format(cls)
            filename = self._get_vehicles_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index['filename'], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_vehicles_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_vehicles_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id
    

    def _do_python_eval(self, output_dir = 'output'):
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_vehicles_results_file_template().format(cls)
            rec, prec, ap = vehicles_eval(
                filename, cls, self._image_index, cachedir,
                ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')






