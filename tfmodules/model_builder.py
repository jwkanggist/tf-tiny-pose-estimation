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


class ModelBuilder(object):

    def __init__(self,model_config):
        self._model_config = model_config

        self.dropout_keeprate = tf.placeholder(dtype=tf.float32)



    def get_model(self, model_in,scope):

        with tf.variable_scope(name_or_scope=scope, values=[model_in]):
        # < complete codes here >

        tf.logging.info('[ModelBuilder] model building complete')
        tf.logging.info('[ModelBuilder] model output shape = %s'%model_out.shape)
        return model_out







    def _get_reception_layer(self,ch_in,
                            num_outputs,
                            model_config,
                            model_config_separable_conv,
                            scope='recept'):

        with tf.variable_scope(name_or_scope=scope,values=[ch_in]):
        # < complete codes here >

        return net






    def _get_output_layer(self,ch_in,
                         num_outputs,
                         model_config,
                         scope='output'):


        with tf.variable_scope(name_or_scope=scope, values=[ch_in]):
        # < complete codes here >

        return out






    def _get_hourglass_layer(self,ch_in,
                            model_config,
                            model_config_separable_conv,
                            scope='hourglass'):

        ch_in_num = ch_in.get_shape().as_list()[3]
        with tf.variable_scope(name_or_scope=scope,values=[ch_in]):

        # < complete codes here >

        return net





    def downsample_hourglass(self,ch_in,
                             model_config,
                             model_config_separable_conv,
                             scope='downsample'):

        ch_in_num   = ch_in.get_shape().as_list()[3]
        with tf.variable_scope(name_or_scope=scope,values=[ch_in]):
        # < complete codes here >

        return net






    def upsample_hourglass(self,ch_in,
                                model_config,
                                model_config_separable_conv,
                                scope='upsample'):

        input_shape     = ch_in.get_shape().as_list()
        output_shape    = [int(input_shape[1]*model_config.updown_rate),
                           int(input_shape[2]*model_config.updown_rate)]

        ch_in_num   = input_shape[3]
        with tf.variable_scope(name_or_scope=scope,values=[ch_in]):
        # < complete codes here >

        return net






    def _get_separable_conv2d(self,ch_in,
                                ch_out_num,
                                model_config,
                                scope='separable_conv'):

        conv2d_padding = 'SAME'
        net = ch_in
        kernel_size  = model_config.kernel_shape_dwise
        stride       = model_config.stride_dwise

        with tf.variable_scope(name_or_scope=scope, default_name='separable_conv2d', values=[ch_in]):
            with slim.arg_scope([model_config.normalizer_fn],
                                decay       =model_config.batch_norm_decay,
                                fused       =model_config.batch_norm_fused,
                                is_training =model_config.is_trainable,
                                activation_fn=None):
                '''
                    Note that "slim.separable_convolution2cd with num_outputs == None" 
                    provides equivalent implementation to the depthwise convolution 
                    with ch_in_num == ch_out_num
                '''
                # depthwise conv with 3x3 kernal
                net = slim.separable_convolution2d(inputs               =net,
                                                   num_outputs          =None,
                                                   kernel_size          =kernel_size,
                                                   depth_multiplier     =1.0,
                                                   stride               =[stride, stride],
                                                   padding              =conv2d_padding,
                                                   activation_fn        =None,
                                                   normalizer_fn        =model_config.normalizer_fn,
                                                   biases_initializer   =None,
                                                   weights_initializer  =model_config.weights_initializer,
                                                   trainable            =model_config.is_trainable,
                                                   scope                =scope + '_dwise_conv')
                # intermediate activation
                if model_config.activation_fn_dwise is not None:
                    net = model_config.activation_fn_dwise(net)

                # pointwise conv with 1x1 kernal
                net = slim.conv2d(inputs                =net,
                                  num_outputs           =ch_out_num,
                                  kernel_size           =[1, 1],
                                  stride                =[1, 1],
                                  padding               ='SAME',
                                  activation_fn         =None,
                                  normalizer_fn         =model_config.normalizer_fn,
                                  biases_initializer    =None,
                                  weights_initializer   =model_config.weights_initializer,
                                  trainable             =model_config.is_trainable,
                                  scope                 =scope + '_pwise_conv')

                # output activation
                if model_config.activation_fn_pwise is not None:
                    net = model_config.activation_fn_pwise(net)

        return net

    # def show_model_shape(self):
