#!/usr/bin/env python

'''
Title           :plot_learning_curve.py
Description     :This script generates learning curves for caffe models
Author          :Adil Moujahid (modified by JK Jung)
Date Created    :2016-06-19
Date Modified   :2017-11-08
version         :0.2
usage           :python ./tools/plot_learning_curve.py ./experiments/logs/XXXXXX.log
python_version  :2.7.x
'''

import os
import sys
import subprocess
import pandas as pd

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt

AVG_INTERVAL = 20

plt.style.use('ggplot')

tools_path = os.path.dirname(os.path.realpath(__file__))
model_log_path = sys.argv[1]
#learning_curve_path = sys.argv[2]

#Get directory where the model logs is saved, and move to it
model_log_dir_path = os.path.dirname(os.path.realpath(model_log_path))
os.chdir(model_log_dir_path)

'''
Generating training and test logs
'''
#Parsing training/validation logs
command = tools_path + '/extra/parse_log.sh ' + os.path.join(model_log_dir_path, os.path.basename(model_log_path))
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
#Read training and test logs
train_log_path = os.path.join(model_log_dir_path, os.path.basename(model_log_path)) + '.train'
#test_log_path = model_log_path + '.test'
train_log = pd.read_csv(train_log_path, delim_whitespace=True)
#test_log = pd.read_csv(test_log_path, delim_whitespace=True)

xx = []
yy = []
for i in range(1, len(train_log['#Iters']), AVG_INTERVAL):
    xx.append(train_log['#Iters'][i])
    yy.append(sum(train_log['TrainingLoss'][i:(i+AVG_INTERVAL-1)]) / AVG_INTERVAL)

'''
Making learning curve
'''
fig, ax1 = plt.subplots()

#Plotting training and test losses
train_loss, = ax1.plot(xx, yy, color='red', alpha=.8)
#train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red', alpha=.8)
#test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=2, color='green')
ax1.set_ylim(ymin=0, ymax=1)
ax1.set_xlabel('Iterations', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.tick_params(labelsize=12)
#Plotting test accuracy
#ax2 = ax1.twinx()
#test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'], linewidth=2, color='blue')
#ax2.set_ylim(ymin=0, ymax=1)
#ax2.set_ylabel('Accuracy', fontsize=15)
#ax2.tick_params(labelsize=15)
#Adding legend
#plt.legend([train_loss, test_loss, test_accuracy], ['Training Loss', 'Test Loss', 'Test Accuracy'],  bbox_to_anchor=(1, 0.8))
plt.legend([train_loss], ['Training Loss'], bbox_to_anchor=(1, 0.8))
plt.title('Training Curve', fontsize=18)
#Saving learning curve
#plt.savefig(learning_curve_path)
plt.show()

'''
Deleting training and test logs
'''
command = 'rm ' + train_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()

#command = command = 'rm ' + test_log_path
#process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
#process.wait()
