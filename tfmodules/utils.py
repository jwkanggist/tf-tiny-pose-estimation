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

from train_config  import TrainConfig
from model_config  import ModelConfig

# config instance generation
train_config    = TrainConfig()
model_config    = ModelConfig()



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




def metric_fn(labels, logits,pck_threshold):
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

    with tf.name_scope('metric_fn',values=[labels, logits,pck_threshold]):

        # get predicted coordinate
        pred_head_xy       = argmax_2d(logits[:,:,:,0:1])
        pred_neck_xy       = argmax_2d(logits[:,:,:,1:2])
        pred_rshoulder_xy  = argmax_2d(logits[:,:,:,2:3])
        pred_lshoulder_xy  = argmax_2d(logits[:,:,:,3:4])

        label_head_xy      = argmax_2d(labels[:,:,:,0:1])
        label_neck_xy      = argmax_2d(labels[:,:,:,1:2])
        label_rshoulder_xy = argmax_2d(labels[:,:,:,2:3])
        label_lshoulder_xy = argmax_2d(labels[:,:,:,3:4])


        # error distance measure
        metric_err_fn                 = train_config.metric_fn

        # distance == root mean square
        head_neck_dist, update_op_head_neck_dist     = metric_err_fn(labels=label_head_xy,
                                                      predictions=label_neck_xy)

        errdist_head,update_op_errdist_head             = metric_err_fn(labels=label_head_xy,
                                                                        predictions=pred_head_xy)
        errdist_neck,update_op_errdist_neck             = metric_err_fn(labels=label_neck_xy,
                                                                        predictions= pred_neck_xy)
        errdist_rshoulder, update_op_errdist_rshoulder  = metric_err_fn(labels=label_rshoulder_xy,
                                                                        predictions= pred_rshoulder_xy)
        errdist_lshoulder, update_op_errdist_lshoulder  = metric_err_fn(labels=label_lshoulder_xy,
                                                                        predictions= pred_lshoulder_xy)
        # percentage of correct keypoints
        total_errdist = (errdist_head +
                         errdist_neck +
                         errdist_rshoulder +
                         errdist_lshoulder) / head_neck_dist

        update_op_total_errdist = (update_op_errdist_head +
                                   update_op_errdist_neck +
                                   update_op_errdist_rshoulder +
                                   update_op_errdist_lshoulder) / update_op_head_neck_dist

        pck =            tf.metrics.percentage_below(values=total_errdist,
                                                   threshold=pck_threshold,
                                                   name=    'pck_' + str(pck_threshold))


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

                            'errdist_lshou': (errdist_lshoulder/head_neck_dist,
                                                    update_op_errdist_lshoulder /update_op_head_neck_dist),
                            'pck': pck
                        }

    return metric_dict




def summary_fn(loss,
               total_out_losssum,
               total_mid_losssum_list,
               learning_rate,
               input_images,
               label_heatmap,
               pred_out_heatmap,
               pred_mid_heatmap):
    '''

        code ref: https://github.com/wookayin/tensorflow-plot
    '''

    tf.summary.scalar(name='loss', tensor=loss, family='outlayer')
    tf.summary.scalar(name='out_loss', tensor=total_out_losssum, family='outlayer')
    tf.summary.scalar(name='learning_rate', tensor=learning_rate, family='outlayer')


    batch_size          = train_config.batch_size
    resized_input_image = tf.image.resize_bicubic(images= input_images,
                                                  size=[int(DEFAULT_HG_INOUT_RESOL),
                                                        int(DEFAULT_HG_INOUT_RESOL)],
                                                  align_corners=False)
    tf.logging.info ('[summary_fn] batch_size = %s' % batch_size)
    tf.logging.info ('[summary_fn] resized_input_image.shape= %s' % resized_input_image.get_shape().as_list())
    tf.logging.info ('[summary_fn] label_heatmap.shape= %s' % label_heatmap.get_shape().as_list())
    tf.logging.info ('[summary_fn] pred_out_heatmap.shape= %s' % pred_out_heatmap.get_shape().as_list())


    if train_config.is_summary_heatmap:
        summary_name_true_heatmap           = "true_heatmap_summary"
        summary_name_pred_out_heatmap       = "pred_out_heatmap_summary"
        summary_name_pred_mid_heatmap       = "pred_mid_heatmap_summary"

        for keypoint_index in range(0,train_config.output_chnum):
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


        for n in range(0, model_config.num_of_hgstacking - 1):
            tf.logging.info ('[summary_fn] pred_mid_heatmap.shape= %s' % pred_mid_heatmap[0].get_shape().as_list())

            tf.summary.scalar(name='mid_loss' + str(n),
                              tensor=total_mid_losssum_list[n],
                              family='midlayer')

            for keypoint_index in range(0,train_config.output_chnum):
                tfplot.summary.plot_many(name       =summary_name_pred_mid_heatmap + '_' +
                                                     str(keypoint_index) +
                                                     '_hgstage'+str(n),
                                         plot_func  =overlay_attention_batch,
                                         in_tensors =[pred_mid_heatmap[n][:, :, :, keypoint_index],
                                                      resized_input_image],
                                         max_outputs=batch_size)

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