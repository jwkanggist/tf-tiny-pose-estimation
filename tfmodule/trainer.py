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
import os


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

from model_builder import ModelBuilder
from data_loader   import DataLoader


def train(dataset_train, dataset_test):
    model_config = ModelConfig()
    train_config = TrainConfig()

    # build dataset ========================

    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_train_iterator = dataset_train.make_one_shot_iterator()
    dataset_test_iterator  = dataset_test.make_one_shot_iterator()

    dataset_iterator = tf.data.Iterator.from_string_handle(dataset_handle,
                                                           dataset_train.output_types,
                                                           dataset_train.output_shapes)
    inputs, true_heatmap =  dataset_iterator.get_next()

    # model building =========================
    with tf.device('/device:GPU:0'):
        # < complete codes here >
        modelbuilder = ModelBuilder(model_config=model_config)
        pred_heatmap = modelbuilder.get_model(model_in=inputs,
                                              scope='model')

    # traning ops =============================================
        # < complete codes here >
        loss_heatmap        = train_config.loss_fn(true_heatmap - pred_heatmap) / train_config.batch_size
        loss_regularizer    = tf.losses.get_regularization_loss()
        loss_op             = loss_heatmap + loss_regularizer


        global_step_op      = tf.train.get_global_step()
        batchnum_per_epoch  = np.floor(train_config.train_data_size / train_config.batch_size)
        current_epoch       = (tf.cast(global_step_op, tf.float32) /
                                batchnum_per_epoch)

        lr_op = tf.train.exponential_decay(learning_rate=train_config.learning_rate,
                                           global_step=global_step_op,
                                           decay_steps=train_config.learning_rate_decay_step,
                                           decay_rate=train_config.learning_rate_decay_rate,
                                           staircase=True)

        opt_op      = train_config.opt_fn(learning_rate=lr_op,name='opt_op')
        train_op    = opt_op.minimize(loss_op, global_step_op)



    # For Tensorboard ===========================================
    file_writer = tf.summary.FileWriter(logdir=train_config.tflogdir)
    file_writer.add_graph(tf.get_default_graph())

    tb_summary_loss_train = tf.summary.scalar('loss_train', loss_op)
    tb_summary_loss_test = tf.summary.scalar('loss_test', loss_op)

    tb_summary_lr   = tf.summary.scalar('learning_rate',lr_op)


    # training ==============================

    init_var = tf.global_variables_initializer()
    print('[train] training_epochs = %s' % train_config.training_epochs)
    print('------------------------------------')

    with tf.Session() as sess:
        # Run the variable initializer
        sess.run(init_var)

        train_handle    = sess.run(dataset_train_iterator.string_handle())
        test_handle     = sess.run(dataset_test_iterator.string_handle())

        rate_record_index = 0

        for epoch in range(train_config.training_epochs):

            train_start_time = time.time()

            # train model
            _,global_step,loss_train = sess.run([train_op,
                                                 global_step_op,
                                                 loss_op],
                                                 feed_dict={dataset_handle: train_handle,
                                                 modelbuilder.dropout_keeprate:model_config.output.dropout_keeprate})


            train_elapsed_time = time.time() - train_start_time


            if train_config.display_step == 0:
                continue
            elif global_step % train_config.display_step == 0:

                # test model
                loss_test = loss_op.eval(feed_dict={dataset_handle: test_handle,
                                                    modelbuilder.dropout_keeprate: 1.0})

                # tf summary
                summary_loss_train = tb_summary_loss_train.eval(feed_dict={dataset_handle: train_handle,
                                                                           modelbuilder.dropout_keeprate:1.0})

                summary_loss_test  = tb_summary_loss_test.eval(feed_dict={dataset_handle: test_handle,
                                                                          modelbuilder.dropout_keeprate: 1.0})

                summary_lr         = tb_summary_lr.eval()

                file_writer.add_summary(summary_loss_train,global_step)
                file_writer.add_summary(summary_loss_test,global_step)
                file_writer.add_summary(summary_lr,global_step)

                print('At step = %d, train elapsed_time = %.1f ms' % (global_step, train_elapsed_time))
                print("Training set loss (avg over batch)= %.2f %%  " % (loss_train))
                print("Test set Err loss (total batch)= %.2f %%" % (loss_test))
                print("--------------------------------------------")


        print("Training finished!")

    file_writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)



    # dataloader instance gen
    dataset_train, dataset_test = \
        [DataLoader(
        is_training     =is_training,
        data_dir        =DATASET_DIR,
        transpose_input =False,
        use_bfloat16    =False) for is_training in [True, False]]


    # model tranining
    with tf.name_scope(name='trainer', values=[dataset_train, dataset_test]):
        # < complete the train() function call >
        train(dataset_train=dataset_train,
              dataset_test=dataset_test)