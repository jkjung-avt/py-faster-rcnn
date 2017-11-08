Steps to train a Faster RCNN model with the Brainwash dataset in DetectNet (KITTI) format

The following steps demonstrate how to train a VGG16 based Faster RCNN:

1. Put the data into the './data/brainwash/detectnet_640x480/' folder, as shown below. Note that the 'annotations_cache' and 'results' folders would be created by test_net.py in a later step.

   data/brainwash
        ├── annotations_cache
        ├── detectnet_640x480
        │   ├── train
        │   │   ├── images
        │   │   └── labels
        │   └── val
        │       ├── images
        │       └── labels
        └── results

2. Download the pre-trained VGG16 weights by './data/scripts/fetch_imagenet_models.sh', as documented by rbgirshick originally.

3. Check and make sure the parameters in the prototxt files are what you intend to be.

   models/brainwash/VGG16/faster_rcnn_end2end/solver.prototxt
   models/brainwash/VGG16/faster_rcnn_end2end/train.prototxt

4. Verify the config file.

   experiments/cfgs/brainwash.yml

5. To run training/testing with the script: (GPUID=0, VGG16)

   $ ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 brainwash

6. To check training loss curve during (or after) training: (example)

   $ ./tools/plot_learning_curve.py ./experiments/logs/faster_rcnn_end2end_VGG16_.txt.2017-10-27_09-56-48

------

The following is for manual training/testing/finetuning.

7. Run the following command to start training. This trains the model for 90,000 iterations. And it writes logs to './experiments/logs/brainwash-1.log'.

   $ time python ./tools/train_net.py --gpu 0 --solver ./models/brainwash/VGG16/faster_rcnn_end2end/solver.prototxt --weights ./data/imagenet_models/VGG16.v2.caffemodel --imdb brainwash_train --iters 90000 --cfg experiments/cfgs/brainwash.yml 2>&1 | tee -a ./experiments/logs/brainwash-1.log

8. To test the trained model (taking snapshot #90,000 as example). The output is the mAP value.

   $ python ./tools/test_net.py --gpu 0 --def ./models/brainwash/VGG16/faster_rcnn_end2end/test.prototxt --net ./output/faster_rcnn_end2end/brainwash_train/vgg16_faster_rcnn_iter_90000.caffemodel --imdb brainwash_val --cfg experiments/cfgs/brainwash.yml

9. If you'd like to further finetune a trained model, you can try this. Note that in 'solver-2.prototxt' uses a different learning rate schedule from 'solver.prototxt'. The following example trains the model for additional 30,000 iterations, and writes the output to ./output/faster_rcnn_end2end/brainwash_train/brainwash_vgg16_finetune_iter_xxxx.caffemodel'.

   $ time python ./tools/train_net.py --gpu 0 --solver ./models/brainwash/VGG16/faster_rcnn_end2end/solver-2.prototxt --weights ./output/faster_rcnn_end2end/brainwash_train/brainwash_vgg16_iter_90000.caffemodel --imdb brainwash_train --iters 30000 --cfg experiments/cfgs/brainwash.yml 2>&1 | tee -a ./experiments/logs/brainwash-2.log

10. To deploy the trained model onto Jetson TX2, you mainly need the trained caffemodel snapshot: './output/faster_rcnn_end2end/brainwash_train/brainwash_vgg16_iter_90000.caffemodel'

   More specifically, make a copy of 'brainwash_vgg16_iter_90000.caffemodel' and put it onto Jetson TX2 under './data/faster_rcnn_models/'. Then put all PNG image files to be tested in the './demo/brainwash' folder. Run the following command, which would randomly pick 10 images for testing.

   $ ./tools/demo_brainwash.sh --net vgg16
