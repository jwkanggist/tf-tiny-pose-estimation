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
# ===================================================================================
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import getcwd
from os import chdir

chdir('../..')
sys.path.insert(0,getcwd())
print ('getcwd() = %s' % getcwd())


import tensorflow as tf

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# image processing tools
import cv2

# custom packages

from path_manager import TF_MODULE_DIR
from path_manager import EXPORT_DIR
from path_manager import COCO_DATALOAD_DIR
from path_manager import DATASET_DIR


sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,EXPORT_DIR)
sys.path.insert(0,COCO_DATALOAD_DIR)


# < here you need import your module >
from model_config import ModelConfig
from train_config import TrainConfig
from train_config import PreprocessingConfig

from data_loader   import DataLoader
from utils import metric_fn
from utils import argmax_2d



IMAGE_MAX_VALUE = 255.0
train_config   = TrainConfig()
model_config   = ModelConfig(setuplog_dir=train_config.setuplog_dir)
preproc_config = PreprocessingConfig(setuplog_dir = train_config.setuplog_dir)

TOP         = 0
NECK        = 1
RSHOULDER   = 2
RELBOW      = 3
RWRIST      = 4
LSHOULDER   = 5
LELBOW      = 6
LWRIST      = 7
RHIP        = 8
RKNEE       = 9
RANKLE      = 10
LHIP        = 11
LKNEE       = 12
LANKLE      = 13

class DataLoaderTest(tf.test.TestCase):

    def test_data_loader_coco(self):
        '''
            This test checks below:
            - whether tfrecord is correctly read
        '''

        # datadir = TFRECORD_TESTIMAGE_DIR
        datadir = DATASET_DIR
        print('\n---------------------------------------------------------')
        print('[test_data_loader_coco] data_dir = %s' % datadir)

        dataset_train = \
            DataLoader(
                is_training=True,
                data_dir=datadir,
                transpose_input=False,
                train_config    = train_config,
                model_config    = model_config,
                preproc_config  = preproc_config,
                use_bfloat16=False)

        dataset = dataset_train
        dataset                     = dataset.input_fn()
        iterator_train              = dataset.make_initializable_iterator()
        feature_op, labels_op       = iterator_train.get_next()

        #------------ lables ------------#
        argmax_2d_top_op            = argmax_2d(tensor=labels_op[:, :, :, 0:1])
        argmax_2d_neck_op           = argmax_2d(tensor=labels_op[:, :, :, 1:2])
        argmax_2d_rshoulder_op      = argmax_2d(tensor=labels_op[:, :, :, 2:3])
        argmax_2d_relbow_op         = argmax_2d(tensor=labels_op[:, :, :, 3:4])

        argmax_2d_rwrist_op         = argmax_2d(tensor=labels_op[:, :, :, 4:5])
        argmax_2d_lshoulder_op      = argmax_2d(tensor=labels_op[:, :, :, 5:6])
        argmax_2d_lelbow_op         = argmax_2d(tensor=labels_op[:, :, :, 6:7])
        argmax_2d_lwrist_op         = argmax_2d(tensor=labels_op[:, :, :, 7:8])

        argmax_2d_rhip_op           = argmax_2d(tensor=labels_op[:, :, :, 8:9])
        argmax_2d_rknee_op          = argmax_2d(tensor=labels_op[:, :, :, 9:10])
        argmax_2d_rankle_op         = argmax_2d(tensor=labels_op[:, :, :, 10:11])
        argmax_2d_lhip_op           = argmax_2d(tensor=labels_op[:, :, :, 11:12])

        argmax_2d_lklee_op          = argmax_2d(tensor=labels_op[:, :, :, 12:13])
        argmax_2d_lankle_op         = argmax_2d(tensor=labels_op[:, :, :, 13:14])

        output_coordinate           = tf.stack([argmax_2d_top_op,
                                                argmax_2d_neck_op,
                                                argmax_2d_rshoulder_op,
                                                argmax_2d_relbow_op,
                                                argmax_2d_rwrist_op,
                                                argmax_2d_lshoulder_op,
                                                argmax_2d_lelbow_op,
                                                argmax_2d_lwrist_op,
                                                argmax_2d_rhip_op,
                                                argmax_2d_rknee_op,
                                                argmax_2d_rankle_op,
                                                argmax_2d_lhip_op,
                                                argmax_2d_lklee_op,
                                                argmax_2d_lankle_op])

        metric_dict_op      = metric_fn(labels=labels_op,
                                        logits=labels_op,
                                        pck_threshold=0.2,
                                        train_config=train_config)


        metric_fn_var       = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope='metric_fn')
        metric_fn_var_init  = tf.variables_initializer(metric_fn_var)

        favorite_image_index = 5

        with self.test_session() as sess:
            sess.run(iterator_train.initializer)

            # init variable used in metric_fn_var_init
            sess.run(metric_fn_var_init)

            for n in range(0,20):
                print ( 'Index n = %s' % n)
                # argmax2d find coordinate of head
                # containing one heatmap
                feature_numpy, labels_numpy, \
                output_coordinate_numpy,\
                metric_dict   \
                    = sess.run([feature_op,
                                labels_op,
                                output_coordinate,
                                metric_dict_op])

                # feature_numpy, labels_numpy, = sess.run([feature_op,labels_op])


                # some post processing
                image_sample          = feature_numpy[favorite_image_index,:,:,:]

                print('[test_data_loader_coco] sum of single label heatmap =%s'% \
                      labels_numpy[favorite_image_index, :, :, 0].sum().sum())

                # 256 to 64
                heatmap_size        = int(model_config._output_size)
                image_sample_resized  = cv2.resize(image_sample.astype(np.uint8),
                                                   dsize=(heatmap_size,
                                                          heatmap_size),
                                                   interpolation=cv2.INTER_CUBIC)
                '''
                    marking the annotation
                    # # keypoint_top[0] : x
                    # # keypoint_top[1] : y
                '''

                keypoint_top        = output_coordinate_numpy[TOP,favorite_image_index].astype(np.uint8)
                keypoint_neck       = output_coordinate_numpy[NECK,favorite_image_index].astype(np.uint8)

                keypoint_rshoulder  = output_coordinate_numpy[RSHOULDER,favorite_image_index].astype(np.uint8)
                keypoint_lshoulder  = output_coordinate_numpy[LSHOULDER,favorite_image_index].astype(np.uint8)

                keypoint_rknee      = output_coordinate_numpy[RKNEE,favorite_image_index].astype(np.uint8)
                keypoint_lknee      = output_coordinate_numpy[LKNEE,favorite_image_index].astype(np.uint8)

                keypoint_rankle     = output_coordinate_numpy[RANKLE,favorite_image_index].astype(np.uint8)
                keypoint_lankle     = output_coordinate_numpy[LANKLE,favorite_image_index].astype(np.uint8)

                image_sample_resized[keypoint_top[1],keypoint_top[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_top[1],keypoint_top[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_top[1],keypoint_top[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_neck[1],keypoint_neck[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_neck[1],keypoint_neck[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_neck[1],keypoint_neck[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_rankle[1],keypoint_rankle[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rankle[1],keypoint_rankle[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rankle[1],keypoint_rankle[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_lankle[1],keypoint_lankle[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lankle[1],keypoint_lankle[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lankle[1],keypoint_lankle[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_rknee[1],keypoint_rknee[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rknee[1],keypoint_rknee[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rknee[1],keypoint_rknee[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_lknee[1],keypoint_lknee[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lknee[1],keypoint_lknee[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_lknee[1],keypoint_lknee[0],2] = IMAGE_MAX_VALUE

                print ('[test_data_loader_coco] keypoint_top       = (%s,%s)' % (keypoint_top[0],keypoint_top[1]))
                print ('[test_data_loader_coco] keypoint_neck      = (%s,%s)' % (keypoint_neck[0],keypoint_neck[1]))
                print ('[test_data_loader_coco] keypoint_rshoulder = (%s,%s)' % (keypoint_rshoulder[0],keypoint_rshoulder[1]))
                print ('[test_data_loader_coco] keypoint_lshoulder = (%s,%s)' % (keypoint_lshoulder[0],keypoint_lshoulder[1]))
                print ('[test_data_loader_coco] keypoint_rknee   = (%s,%s)' % (keypoint_rknee[0],keypoint_rknee[1]))
                print ('[test_data_loader_coco] keypoint_lknee   = (%s,%s)' % (keypoint_lknee[0],keypoint_lknee[1]))
                print ('[test_data_loader_coco] keypoint_rankle = (%s,%s)' % (keypoint_rankle[0],keypoint_rankle[1]))
                print ('[test_data_loader_coco] keypoint_lankle = (%s,%s)' % (keypoint_lankle[0],keypoint_lankle[1]))
                print (metric_dict)
                print('---------------------------------------------------------')


                plt.figure(1)
                plt.imshow(feature_numpy[favorite_image_index].astype(np.uint8))
                plt.show()

                plt.figure(2)
                plt.imshow(image_sample_resized.astype(np.uint8))
                plt.show()

                #-----------
                # labels_top_numpy        = labels_numpy[favorite_image_index, :, :, TOP] \
                #                           * IMAGE_MAX_VALUE
                # labels_neck_numpy       = labels_numpy[favorite_image_index, :, :, NECK] \
                #                           * IMAGE_MAX_VALUE
                # labels_rshoulder_numpy  = labels_numpy[favorite_image_index, :, :, RSHOULDER] \
                #                           * IMAGE_MAX_VALUE
                # labels_lshoulder_numpy  = labels_numpy[favorite_image_index, :, :, LSHOULDER] \
                #                           * IMAGE_MAX_VALUE
                # labels_rankle_numpy       = labels_numpy[favorite_image_index, :, :, RANKLE] \
                #                           * IMAGE_MAX_VALUE
                # labels_lankle_numpy       = labels_numpy[favorite_image_index, :, :, LANKLE] \
                #                           * IMAGE_MAX_VALUE
                ### heatmaps
                # plt.figure(3)
                # plt.imshow(labels_top_numpy.astype(np.uint8))
                # plt.title('TOP')
                # plt.show()
                #
                # plt.figure(4)
                # plt.imshow(labels_neck_numpy.astype(np.uint8))
                # plt.title('NECK')
                # plt.show()
                #
                # plt.figure(5)
                # plt.imshow(labels_rshoulder_numpy.astype(np.uint8))
                # plt.title('Rshoulder')
                # plt.show()
                #
                # plt.figure(6)
                # plt.imshow(labels_lshoulder_numpy.astype(np.uint8))
                # plt.title('Lshoulder')
                # plt.show()

if __name__ == '__main__':
    tf.test.main()

