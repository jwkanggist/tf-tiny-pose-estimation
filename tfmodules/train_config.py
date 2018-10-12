# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from datetime import datetime
from os import getcwd


class TrainConfig(object):


    def __init__(self):

        self.learning_rate              = 1e-3
        self.learning_rate_decay_step   = 2000
        self.learning_rate_decay_rate   = 0.95
        self.opt_fn                 = tf.train.AdamOptimizer
        self.loss_fn                = tf.nn.l2_loss
        self.batch_size             = 8
        self.metric_fn              = tf.metrics.root_mean_squared_error


        # the number of step between evaluation
        self.display_step   = 50
        self.train_data_size      = 3000
        self.test_data_size       = 1500

        self.training_epochs = int(float(self.train_data_size/self.batch_size) * 10.0)


        self.multiprocessing_num = 1
        self.random_seed         = 66478

        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = getcwd() + '/export/'

        self.ckptdir  = self.root_logdir + '/pb_and_ckpt/'
        self.tflogdir = "{}/run-{}/".format(self.root_logdir+'/tf_logs', now)





class PreprocessingConfig(object):

    def __init__(self):
        # image pre-processing
        self.is_crop                    = True
        self.is_rotate                  = True
        self.is_flipping                = True
        self.is_scale                   = True
        self.is_resize_shortest_edge    = True

        # this is when classification task
        # which has an input as pose coordinate
        # self.is_label_coordinate_norm   = False

        # for ground true heatmap generation
        self.heatmap_std        = 6.0

        self.MIN_AUGMENT_ROTATE_ANGLE_DEG = -15.0
        self.MAX_AUGMENT_ROTATE_ANGLE_DEG = 15.0

        # For normalize the image to zero mean and unit variance.
        self.MEAN_RGB = [0.485, 0.456, 0.406]
        self.STDDEV_RGB = [0.229, 0.224, 0.225]


    def show_info(self):
        tf.logging.info('------------------------')
        tf.logging.info('[train_config] Use is_crop: %s'        % str(self.is_crop))
        tf.logging.info('[train_config] Use is_rotate  : %s'    % str(self.is_rotate))
        tf.logging.info('[train_config] Use is_flipping: %s'    % str(self.is_flipping))
        tf.logging.info('[train_config] Use is_scale: %s'       % str(self.is_scale))
        tf.logging.info('[train_config] Use is_resize_shortest_edge: %s' % str(self.is_resize_shortest_edge))

        if self.is_rotate:

            tf.logging.info('[train_config] MIN_ROTATE_ANGLE_DEG: %s' % str(self.MIN_AUGMENT_ROTATE_ANGLE_DEG))
            tf.logging.info('[train_config] MAX_ROTATE_ANGLE_DEG: %s' % str(self.MAX_AUGMENT_ROTATE_ANGLE_DEG))
        tf.logging.info('[train_config] Use heatmap_std: %s'    % str(self.heatmap_std))
        tf.logging.info('------------------------')

