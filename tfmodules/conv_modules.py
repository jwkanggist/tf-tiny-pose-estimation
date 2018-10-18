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
import numpy as np




def get_separable_conv2d(ch_in,
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
                                               weights_regularizer  =model_config.weights_regularizer,
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
                              weights_regularizer   =model_config.weights_regularizer,
                              trainable             =model_config.is_trainable,
                              scope                 =scope + '_pwise_conv')

            # output activation
            if model_config.activation_fn_pwise is not None:
                net = model_config.activation_fn_pwise(net)

    return net






def get_linear_bottleneck_module(ch_in,
                                ch_out_num,
                                model_config,
                                scope='linear_bottleneck'):

    conv2d_padding = 'SAME'
    net = ch_in
    kernel_size  = model_config.kernel_shape_dwise
    stride       = model_config.stride_dwise

    with tf.variable_scope(name_or_scope=scope,default_name='linear_bottleneck',values=[ch_in]):


        with slim.arg_scope([slim.conv2d],
                            kernel_size         =[1, 1],
                            stride              =[1, 1],
                            padding             ='SAME',
                            activation_fn       =None,
                            weights_initializer =model_config.weights_initializer,
                            weights_regularizer =None,
                            trainable           =model_config.is_trainable):

            # batch_norm w/ relu6 activation
            with slim.arg_scope([model_config.normalizer_fn],
                                decay           = model_config.batch_norm_decay,
                                fused           = model_config.batch_norm_fused,
                                is_training     = model_config.is_trainable,
                                activation_fn   = model_config.activation_fn_pwise):
                '''
                    Note that "slim.separable_convolution2cd with num_outputs == None" 
                    provides equivalent implementation to the depthwise convolution 
                    with ch_in_num == ch_out_num
                '''
                # depthwise conv with 3x3 conv
                # followed by batch_norm and relu6
                net = slim.separable_convolution2d(inputs=              net,
                                                   num_outputs=         None,
                                                   kernel_size=         kernel_size,
                                                   depth_multiplier=    1.0,
                                                   stride=              stride,
                                                   padding=             conv2d_padding,
                                                   activation_fn=       model_config.activation_fn_dwise,
                                                   normalizer_fn=       model_config.normalizer_fn,
                                                   biases_initializer=  None,
                                                   weights_initializer= model_config.weights_initializer,
                                                   weights_regularizer= model_config.weights_regularizer,
                                                   trainable=           model_config.is_trainable,
                                                   scope=               scope + '_dwise_conv')

                # pointwise conv with 1x1 kernal
                # followed by batch_norm and relu
                net_shape   = net.get_shape().as_list()
                net_ch_num  = net_shape[3]

                net = slim.conv2d(inputs=               net,
                                  num_outputs=          net_ch_num,
                                  normalizer_fn=        model_config.normalizer_fn,
                                  biases_initializer=   None,
                                  scope=                scope + '_pwise_conv')

                # linear bottleneck block by conv1x1
                # followed by batch_norm
                net = slim.conv2d(inputs=               net,
                                  num_outputs=          ch_out_num,
                                  normalizer_fn=        None,
                                  biases_initializer=   model_config.biases_initializer,
                                  scope=                scope + '_bottleneck')

                net = model_config.normalizer_fn(inputs=        net,
                                                 activation_fn= None)



    return net








def get_inverted_bottleneck(ch_in,
                             ch_out_num,
                             model_config,
                             scope='inverted_bottleneck'):

    conv2d_padding = 'SAME'
    net = ch_in
    kernel_size  = model_config.kernel_shape_dwise
    stride       = model_config.stride_dwise

    # number of input channel
    ch_in_num       = ch_in.get_shape().as_list()[3]
    expand_ch_num   = np.floor(ch_in_num * model_config.invbottle_expansion_rate)
    with tf.variable_scope(name_or_scope=scope, default_name='inverted_bottleneck', values=[ch_in]):

        with slim.arg_scope([slim.conv2d],
                            kernel_size         =[1, 1],
                            stride              =[1, 1],
                            padding             ='SAME',
                            activation_fn       =None,
                            weights_initializer =model_config.weights_initializer,
                            weights_regularizer =None,
                            trainable           =model_config.is_trainable):

            with slim.arg_scope([model_config.normalizer_fn],
                                decay=model_config.batch_norm_decay,
                                fused=model_config.batch_norm_fused,
                                is_training=model_config.is_trainable,
                                activation_fn=model_config.activation_fn_pwise):

                # linear bottleneck by conv 1x1
                # followed by batch_norm and relu

                net = slim.conv2d(inputs=net,
                                  num_outputs=expand_ch_num,
                                  normalizer_fn=model_config.normalizer_fn,
                                  biases_initializer=None,
                                  scope=scope + '_bottleneck')

                '''
                    Note that "slim.separable_convolution2cd with num_outputs == None" 
                    provides equivalent implementation to the depthwise convolution 
                    with ch_in_num == ch_out_num
                '''
                # depthwise conv with 3x3 conv
                # followed by batch_norm and relu6
                net = slim.separable_convolution2d(inputs=net,
                                                   num_outputs=None,
                                                   kernel_size=kernel_size,
                                                   depth_multiplier=1.0,
                                                   stride=[stride, stride],
                                                   padding=conv2d_padding,
                                                   activation_fn=None,
                                                   normalizer_fn=model_config.normalizer_fn,
                                                   biases_initializer=None,
                                                   weights_initializer=model_config.weights_initializer,
                                                   weights_regularizer=model_config.weights_regularizer,
                                                   trainable=model_config.is_trainable,
                                                   scope=scope + '_dwise_conv')

                # pointwise conv with 1x1 kernal
                # followed by batch_norm
                net = slim.conv2d(inputs=net,
                                  num_outputs=ch_out_num,
                                  normalizer_fn=None,
                                  biases_initializer=model_config.biases_initializer,
                                  scope=scope + '_pwise_conv')

                net = model_config.normalizer_fn(inputs=net,
                                                 activation_fn=None)

            ch_in_shape = ch_in.get_shape().as_list()
            ch_in_num = ch_in_shape[3]

            # shortcut connection
            if ch_in_num != ch_out_num:
                shortcut_out = slim.conv2d(inputs=ch_in,
                                           num_outputs=ch_out_num,
                                           normalizer_fn=None,
                                           biases_initializer=model_config.biases_initializer,
                                           scope=scope + '_shortcut_conv1x1')
            else:
                shortcut_out = ch_in

            if stride > 1:
                shortcut_out = slim.max_pool2d(inputs=shortcut_out,
                                               kernel_size=[3 ,3],
                                               stride=[stride, stride],
                                               padding=conv2d_padding,
                                               scope=scope + '_shortcut_maxpool')

        # elementwise sum for shortcut connection
        net = tf.add(x=shortcut_out,
                     y=net,
                     name=scope + '_out')


    return net
