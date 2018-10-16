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
import numpy as np

import tfplot
import tfplot.summary


def argmax_2d(tensor):

    # input format: BxHxWxD
    assert len(tensor.get_shape()) == 4

    with tf.name_scope(name='argmax_2d',values=[tensor]):
        tensor_shape = tensor.get_shape().as_list()

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tensor_shape[0], -1, tensor_shape[3]))

        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.float32)

        # convert indexes into 2D coordinates
        argmax_x = argmax % tensor_shape[2]
        argmax_y = argmax // tensor_shape[2]

    return tf.concat((argmax_x, argmax_y), axis=1)




def metric_fn(labels, logits,pck_threshold,train_config):
    """Evaluation metric function. Evaluates accuracy.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the model
    to the `metric_fn`, provide as part of the `eval_metrics`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `eval_metrics`.

    Args:
    labels: `Tensor` of labels_heatmap_list
    logits: `Tensor` of logits_heatmap_list

    Returns:
    A dict of the metrics to return from evaluation.
    """
    '''
        Top = 0
        Neck = 1
        RShoulder = 2
        RElbow = 3
        
        RWrist = 4
        LShoulder = 5
        LElbow = 6
        LWrist = 7
        
        RHip = 8
        RKnee = 9
        RAnkle = 10
        LHip = 11
        
        LKnee = 12
        LAnkle = 13
    '''
    with tf.name_scope('metric_fn',values=[labels, logits,pck_threshold]):
        # get predicted coordinate
        pred_head_xy        = argmax_2d(logits[:,:,:,0:1])
        pred_neck_xy        = argmax_2d(logits[:,:,:,1:2])
        pred_rshoulder_xy   = argmax_2d(logits[:,:,:,2:3])
        pred_relbow_xy      = argmax_2d(logits[:,:,:,3:4])

        pred_rwrist_xy      = argmax_2d(logits[:,:,:,4:5])
        pred_lshoulder_xy   = argmax_2d(logits[:,:,:,5:6])
        pred_lelbow_xy      = argmax_2d(logits[:,:,:,6:7])
        pred_lwrist_xy      = argmax_2d(logits[:,:,:,7:8])

        pred_rhip_xy        = argmax_2d(logits[:,:,:,8:9])
        pred_rknee_xy       = argmax_2d(logits[:,:,:,9:10])
        pred_rankle_xy      = argmax_2d(logits[:,:,:,10:11])
        pred_lhip_xy        = argmax_2d(logits[:,:,:,11:12])

        pred_lknee_xy       = argmax_2d(logits[:,:,:,12:13])
        pred_lankle_xy      = argmax_2d(logits[:,:,:,13:14])



        label_head_xy       = argmax_2d(labels[:,:,:,0:1])
        label_neck_xy       = argmax_2d(labels[:,:,:,1:2])
        label_rshoulder_xy  = argmax_2d(labels[:,:,:,2:3])
        label_relbow_xy     = argmax_2d(labels[:,:,:,3:4])

        label_rwrist_xy     = argmax_2d(labels[:,:,:,4:5])
        label_lshoulder_xy  = argmax_2d(labels[:,:,:,5:6])
        label_lelbow_xy     = argmax_2d(labels[:,:,:,6:7])
        label_lwrist_xy     = argmax_2d(labels[:,:,:,7:8])

        label_rhip_xy       = argmax_2d(labels[:,:,:,8:9])
        label_rknee_xy      = argmax_2d(labels[:,:,:,9:10])
        label_rankle_xy     = argmax_2d(labels[:,:,:,10:11])
        label_lhip_xy       = argmax_2d(logits[:,:,:,11:12])

        label_lknee_xy      = argmax_2d(logits[:,:,:,12:13])
        label_lankle_xy     = argmax_2d(logits[:,:,:,13:14])

        # error distance measure
        metric_err_fn                 = train_config.metric_fn

        # distance == root mean square
        head_neck_dist, update_op_head_neck_dist     = metric_err_fn(labels=label_head_xy,
                                                      predictions=label_neck_xy)

        errdist_head,update_op_errdist_head             = metric_err_fn(labels      =label_head_xy,
                                                                        predictions =pred_head_xy)
        errdist_neck,update_op_errdist_neck             = metric_err_fn(labels      =label_neck_xy,
                                                                        predictions =pred_neck_xy)
        errdist_rshoulder,  update_op_errdist_rshoulder  = metric_err_fn(labels     =label_rshoulder_xy,
                                                                        predictions =pred_rshoulder_xy)
        errdist_relbow,     update_op_errdist_relbow     = metric_err_fn(labels     =label_relbow_xy,
                                                                        predictions =pred_relbow_xy)

        errdist_rwrist,     update_op_errdist_rwrist      = metric_err_fn(labels     =label_rwrist_xy,
                                                                         predictions=pred_rwrist_xy)
        errdist_lshoulder,  update_op_errdist_lshoulder  = metric_err_fn(labels     =label_lshoulder_xy,
                                                                        predictions =pred_lshoulder_xy)
        errdist_lelbow,     update_op_errdist_lelbow     = metric_err_fn(labels     =label_lelbow_xy,
                                                                        predictions =pred_lelbow_xy)
        errdist_lwrist,     update_op_errdist_lwrist     = metric_err_fn(labels     =label_lwrist_xy,
                                                                        predictions =pred_lwrist_xy)


        errdist_rhip,       update_op_errdist_rhip       = metric_err_fn(labels     =label_rhip_xy,
                                                                        predictions =pred_rhip_xy)
        errdist_rknee,      update_op_errdist_rknee      = metric_err_fn(labels     =label_rknee_xy,
                                                                         predictions=pred_rknee_xy)
        errdist_rankle,     update_op_errdist_rankle     = metric_err_fn(labels     =label_rankle_xy,
                                                                         predictions=pred_rankle_xy)
        errdist_lhip,       update_op_errdist_lhip       = metric_err_fn(labels     =label_lhip_xy,
                                                                         predictions=pred_lhip_xy)

        errdist_lknee,       update_op_errdist_lknee     = metric_err_fn(labels     =label_lknee_xy,
                                                                        predictions =pred_lknee_xy)
        errdist_lankle,     update_op_errdist_lankle     = metric_err_fn(labels     =label_lankle_xy,
                                                                        predictions =pred_lankle_xy)

        # percentage of correct keypoints
        total_errdist = (errdist_head +
                         errdist_neck +
                         errdist_rshoulder +
                         errdist_relbow +
                         errdist_rwrist    +
                         errdist_lshoulder +
                         errdist_lelbow    +
                         errdist_lwrist    +
                         errdist_rhip      +
                         errdist_rknee     +
                         errdist_rankle    +
                         errdist_lhip      +
                         errdist_lknee      +
                         errdist_lankle    ) / head_neck_dist

        update_op_total_errdist = (update_op_errdist_head +
                                   update_op_errdist_neck +
                                   update_op_errdist_rshoulder +
                                   update_op_errdist_relbow +
                                   update_op_errdist_rwrist    +
                                   update_op_errdist_lshoulder +
                                   update_op_errdist_lelbow    +
                                   update_op_errdist_lwrist    +
                                   update_op_errdist_rhip      +
                                   update_op_errdist_rknee     +
                                   update_op_errdist_rankle    +
                                   update_op_errdist_lhip      +
                                   update_op_errdist_lknee      +
                                   update_op_errdist_lankle) / update_op_head_neck_dist

        pck =            tf.metrics.percentage_below(values=total_errdist,
                                                   threshold=pck_threshold,
                                                   name=    'pck_' + str(pck_threshold))
        '''
            Top = 0
            Neck = 1
            RShoulder = 2
            RElbow = 3

            RWrist = 4
            LShoulder = 5
            LElbow = 6
            LWrist = 7

            RHip = 8
            RKnee = 9
            RAnkle = 10
            LHip = 11

            LKnee = 12
            LAnkle = 13
        '''
        # form a dictionary
        metric_dict = {
                            'label_head_neck_dist' : (head_neck_dist/head_neck_dist,
                                                      update_op_head_neck_dist/update_op_head_neck_dist),

                            'total_errdis': (total_errdist,update_op_total_errdist),

                            'errdist_head': (errdist_head/head_neck_dist,
                                                    update_op_errdist_head/update_op_head_neck_dist),
                            'errdist_neck': (errdist_neck/head_neck_dist,
                                                    update_op_errdist_neck/update_op_head_neck_dist),
                            'errdist_rshou': (errdist_rshoulder/head_neck_dist,
                                                    update_op_errdist_rshoulder /update_op_head_neck_dist),
                            'errdist_relbow': (errdist_relbow / head_neck_dist,
                                               update_op_errdist_relbow / update_op_head_neck_dist),


                            'errdist_rwrist': (errdist_rwrist / head_neck_dist,
                                                     update_op_errdist_rwrist / update_op_head_neck_dist),
                            'errdist_lshou': (errdist_lshoulder/head_neck_dist,
                                                    update_op_errdist_lshoulder /update_op_head_neck_dist),
                            'errdist_lelbow': (errdist_lelbow / head_neck_dist,
                                                    update_op_errdist_lelbow / update_op_head_neck_dist),
                            'errdist_lwrist': (errdist_lwrist/head_neck_dist,
                                                    update_op_errdist_lwrist /  update_op_head_neck_dist),


                            'errdist_rhip': (errdist_rhip / head_neck_dist,
                                                    update_op_errdist_rhip /  update_op_head_neck_dist),
                            'errdist_rknee': (errdist_rknee / head_neck_dist,
                                                    update_op_errdist_rknee / update_op_head_neck_dist),
                            'errdist_rankle': (errdist_rankle / head_neck_dist,
                                                    update_op_errdist_rankle / update_op_head_neck_dist),
                            'errdist_lhip' :  (errdist_lhip / head_neck_dist,
                                                    update_op_errdist_lhip / update_op_head_neck_dist),


                            'errdist_lknee' : (errdist_lknee / head_neck_dist,
                                                    update_op_errdist_lknee / update_op_head_neck_dist),
                            'errdist_lankle': (errdist_lankle / head_neck_dist,
                                                    update_op_errdist_lankle / update_op_head_neck_dist),
                            'pck': pck
                        }

    return metric_dict




def summary_fn(loss,
               total_out_losssum,
               learning_rate,
               input_images,
               label_heatmap,
               pred_out_heatmap,
               train_config,
               model_config):
    '''

        code ref: https://github.com/wookayin/tensorflow-plot
    '''

    tf.summary.scalar(name='loss', tensor=loss)
    tf.summary.scalar(name='out_loss', tensor=total_out_losssum)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)


    batch_size          = train_config.batch_size
    resized_input_image = tf.image.resize_bicubic(images= input_images,
                                                  size=[model_config._output_size,
                                                        model_config._output_size],
                                                  align_corners=False)
    tf.logging.info ('[summary_fn] batch_size = %s' % batch_size)
    tf.logging.info ('[summary_fn] resized_input_image.shape= %s' % resized_input_image.get_shape().as_list())
    tf.logging.info ('[summary_fn] label_heatmap.shape= %s' % label_heatmap.get_shape().as_list())
    tf.logging.info ('[summary_fn] pred_out_heatmap.shape= %s' % pred_out_heatmap.get_shape().as_list())


    if train_config.is_summary_heatmap:
        summary_name_true_heatmap           = "true_heatmap_summary"
        summary_name_pred_out_heatmap       = "pred_out_heatmap_summary"

        for keypoint_index in range(0,model_config.output_chnum):
            tfplot.summary.plot_many(name           =summary_name_true_heatmap + '_' +
                                                     str(keypoint_index),
                                     plot_func      =overlay_attention_batch,
                                     in_tensors     =[label_heatmap[:,:,:,keypoint_index],
                                                      resized_input_image],
                                     max_outputs    =batch_size)

            tfplot.summary.plot_many(name           =summary_name_pred_out_heatmap + '_' +
                                                     str(keypoint_index),
                                     plot_func      =overlay_attention_batch,
                                     in_tensors     =[pred_out_heatmap[:,:,:,keypoint_index],
                                                      resized_input_image],
                                     max_outputs    =batch_size)


    return tf.summary.merge_all()






def overlay_attention_batch(attention, image,
                            alpha=0.5, cmap='jet'):

    fig = tfplot.Figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    # fig.subplots_adjust(0, 0, 1, 1)  # get rid of margins

    # print (attention.shape)
    # print (image.shape)
    # print ('[tfplot] attention  =%s' % attention)
    # print ('[tfplot] image      =%s' % image)
    image = image.astype(np.uint8)
    H, W = attention.shape
    ax.imshow(image, extent=[0, H, 0, W])
    ax.imshow(attention, cmap=cmap,
              alpha=alpha, extent=[0, H, 0, W])

    return fig