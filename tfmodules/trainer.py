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

import tensorflow as tf
import numpy as np
import time
import sys

from path_manager import TF_MODULE_DIR
from path_manager import EXPORT_DIR
from path_manager import COCO_DATALOAD_DIR
from path_manager import DATASET_DIR


sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,EXPORT_DIR)
sys.path.insert(0,COCO_DATALOAD_DIR)


from model_config import ModelConfig

from train_config import TrainConfig
from train_config import PreprocessingConfig

from model_builder import ModelBuilder
from data_loader   import DataLoader
from utils         import summary_fn


def train(dataset_train, dataset_valid):
    model_config    = ModelConfig()

    train_config    = TrainConfig()
    preproc_config  = PreprocessingConfig()


    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_train_iterator  = dataset_train.make_one_shot_iterator()
    dataset_valid_iterator  = dataset_valid.make_one_shot_iterator()


    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_iterator = tf.data.Iterator.from_string_handle(dataset_handle,
                                                           dataset_train.output_types,
                                                           dataset_train.output_shapes)
    inputs, true_heatmap =  dataset_iterator.get_next()


    # model building =========================
    # < complete codes here >
    modelbuilder = ModelBuilder(model_config=model_config)
    pred_heatmap = modelbuilder.get_model(model_in=inputs,
                                          scope='model')

    # traning ops =============================================
    # < complete codes here >
    loss_heatmap_op        = train_config.loss_fn(true_heatmap - pred_heatmap) / train_config.batch_size
    loss_regularizer_op    = tf.losses.get_regularization_loss()
    loss_op                = loss_heatmap_op + loss_regularizer_op

    global_step = tf.Variable(0, trainable=False)
    batchnum_per_epoch  = np.floor(train_config.train_data_size / train_config.batch_size)

    lr_op = tf.train.exponential_decay(learning_rate=train_config.learning_rate,
                                       global_step=global_step,
                                       decay_steps=train_config.learning_rate_decay_step,
                                       decay_rate=train_config.learning_rate_decay_rate,
                                       staircase=True)

    opt_op      = train_config.opt_fn(learning_rate=lr_op,name='opt_op')
    train_op    = opt_op.minimize(loss_op, global_step)

    # For Tensorboard ===========================================
    file_writer_train = tf.summary.FileWriter(logdir=train_config.tflogdir +'/train')
    file_writer_valid = tf.summary.FileWriter(logdir=train_config.tflogdir +'/valid')

    file_writer_train.add_graph(tf.get_default_graph())

    # tb_summary_loss= tf.summary.scalar('loss', loss_heatmap_op)
    # tb_summary_lr   = tf.summary.scalar('learning_rate',lr_op)
    # write_op        = tf.summary.merge_all()

    write_op = summary_fn(loss=loss_heatmap_op,
                          total_out_losssum = loss_op,
                          learning_rate     = lr_op,
                          input_images      = inputs,
                          label_heatmap     = true_heatmap,
                          pred_out_heatmap  = pred_heatmap)

    # training ==============================

    init_var = tf.global_variables_initializer()
    saver    = tf.train.Saver()
    print('[train] training_epochs = %s' % train_config.training_epochs)
    print('------------------------------------')

    sess_config = tf.ConfigProto(log_device_placement=True,
                                 gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config = sess_config) as sess:
        # Run the variable initializer
        sess.run(init_var)

        # save graph in pb file
        tf.train.write_graph(sess.graph_def,train_config.ckpt_dir,'model.pb')
        train_handle     = sess.run(dataset_train_iterator.string_handle())
        valid_handle     = sess.run(dataset_valid_iterator.string_handle())

        for epoch in range(train_config.training_epochs):

            train_start_time = time.time()

            # train model
            _,loss_train = sess.run([train_op,loss_op],
                                     feed_dict={dataset_handle: train_handle,
                                     modelbuilder.dropout_keeprate:model_config.output.dropout_keeprate})

            train_elapsed_time  = time.time() - train_start_time
            global_step_eval    = global_step.eval()

            if train_config.display_step == 0:
                continue
            elif global_step_eval % train_config.display_step == 0:
                print('[train] curr epochs = %s' % epoch)

                # # valid model
                loss_train = loss_heatmap_op.eval(feed_dict={dataset_handle: train_handle,
                                                             modelbuilder.dropout_keeprate: 1.0})
                loss_valid = loss_heatmap_op.eval( feed_dict={dataset_handle: valid_handle,
                                                              modelbuilder.dropout_keeprate: 1.0})

                # tf summary
                summary_train = write_op.eval(feed_dict={dataset_handle: train_handle,
                                                        modelbuilder.dropout_keeprate:1.0})
                file_writer_train.add_summary(summary_train, global_step_eval)
                file_writer_train.flush()

                summary_valid  = write_op.eval(feed_dict={dataset_handle: valid_handle,
                                                        modelbuilder.dropout_keeprate: 1.0})
                file_writer_valid.add_summary(summary_valid,global_step_eval)
                file_writer_valid.flush()

                print('At step = %d, train elapsed_time = %.1f ms' % (global_step_eval, train_elapsed_time))
                print("Training set loss (avg over batch)= %.2f   " % (loss_train))
                print("valid set Err loss (total batch)= %.2f %%" % (loss_valid))
                print("--------------------------------------------")

            if global_step_eval % train_config.ckpt_step == 0:
                ckpt_save_path = saver.save(sess,train_config.ckpt_dir +'model.ckpt',global_step=global_step_eval)
                tf.logging.info("Global step - %s: Model saved in file: %s" % (global_step_eval, ckpt_save_path))

        print("Training finished!")

    file_writer_train.close()
    file_writer_valid.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)


    # dataloader instance gen
    dataloader_train, dataloader_valid = \
        [DataLoader(
        is_training     =is_training,
        data_dir        =DATASET_DIR,
        transpose_input =False,
        use_bfloat16    =False) for is_training in [True, False]]


    dataset_train   = dataloader_train.input_fn()
    dataset_valid   = dataloader_valid.input_fn()

    # model training
    with tf.name_scope(name='trainer'):
        # < complete the train() function call >
        train(dataset_train=dataset_train,
              dataset_valid=dataset_valid)

