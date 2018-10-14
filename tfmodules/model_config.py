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
import tensorflow.contrib.slim as slim
from train_config import TrainConfig
import json




class ModelConfig(object):

    def __init__(self):

        self.reception = RecepConfig()
        self.hourglass = HourglassConfig()
        self.output    = OutputConfig()
        self.separable_conv = SeparableConfig()

        self._input_size   = 256
        self._output_size  = 64

        self.input_chnum   = 3
        self.output_chnum  = 14 # number of keypoints
        self.channel_num   = 32

        self.dtype = tf.float32

        # model config logging
        self.model_config_dict      = self.__dict__
        self.reception_config_dict  = self.reception.__dict__
        self.hourglass_config_dict  = self.hourglass.__dict__
        self.output_config_dict     = self.output.__dict__

        train_config = TrainConfig()
        if not tf.gfile.Exists(train_config.setuplog_dir):
            tf.gfile.MakeDirs(train_config.setuplog_dir)

        model_config_filename       = train_config.setuplog_dir + 'model_config.json'
        reception_config_filename   = train_config.setuplog_dir + 'recept_config.json'
        hourglass_config_filename   = train_config.setuplog_dir + 'hourglass_config.json'
        output_config_filename      = train_config.setuplog_dir + 'output_config.json'

        with open(model_config_filename,'w') as fp:
            json.dump(str(self.model_config_dict), fp)

        with open(reception_config_filename,'w') as fp:
            json.dump(str(self.reception_config_dict),fp)

        with open(hourglass_config_filename,'w') as fp:
            json.dump(str(self.hourglass_config_dict),fp)

        with open(output_config_filename,'w') as fp:
            json.dump(str(self.output_config_dict),fp)




class RecepConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True

        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.activation_fn          = tf.nn.relu
        self.normalizer_fn          = slim.batch_norm
        self.is_trainable           = True

        self.kernel_shape ={\
            'r1': [7,7],
            'r4': [3,3]
            }

        self.strides = {\
            'r1': 2,
            'r4': 2
            }





class HourglassConfig(object):

    def __init__(self):
        self.updown_rate            = 2
        self.maxpool_kernel_size    =[3,3]
        self.num_stage              = 3





class OutputConfig(object):

    def __init__(self):
        self.dropout_keeprate       = 0.8
        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.activation_fn          = None
        self.is_trainable           = True

        self.kernel_shape   = [1,1]
        self.stride         = 1





class SeparableConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True

        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.biases_initializer     = slim.init_ops.zeros_initializer()

        self.normalizer_fn          = slim.batch_norm
        self.is_trainable           = True

        self.activation_fn_dwise = None
        self.activation_fn_pwise = tf.nn.relu

        self.kernel_shape_dwise =[3,3]
        self.stride_dwise       = 1

