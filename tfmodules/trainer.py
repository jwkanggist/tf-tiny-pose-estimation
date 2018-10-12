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


# < here you need import your module >
from model_config import ModelConfig
from train_config import TrainConfig

from model_builder import ModelBuilder
from data_loader   import DataLoader


def train(dataset_train, dataset_test):
    model_config = ModelConfig()
    train_config = TrainConfig()

    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_train_iterator = dataset_train.make_one_shot_iterator()
    # dataset_test_iterator  = dataset_test.make_one_shot_iterator()

    inputs = tf.placeholder(dtype=model_config.dtype, shape=[train_config.batch_size,
                                                             model_config._input_size,
                                                             model_config._input_size,
                                                             model_config.input_chnum])

    true_heatmap = tf.placeholder(dtype=model_config.dtype, shape=[train_config.batch_size,
                                                                   model_config._output_size,
                                                                   model_config._output_size,
                                                                   model_config.output_chnum])

    # model building =========================
    with tf.device('/device:CPU:0'):
        # < complete codes here >
        modelbuilder = ModelBuilder(model_config=model_config)
        pred_heatmap = modelbuilder.get_model(model_in=inputs,
                                              scope='model')

    # traning ops =============================================
        # < complete codes here >
        loss_heatmap        = train_config.loss_fn(true_heatmap - pred_heatmap) / train_config.batch_size
        loss_regularizer    = tf.losses.get_regularization_loss()
        loss_op             = loss_heatmap + loss_regularizer

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
    file_writer = tf.summary.FileWriter(logdir=train_config.tflogdir)
    file_writer.add_graph(tf.get_default_graph())

    tb_summary_loss_train = tf.summary.scalar('loss_train', loss_op)
    tb_summary_loss_test = tf.summary.scalar('loss_test', loss_op)

    tb_summary_lr   = tf.summary.scalar('learning_rate',lr_op)


    # training ==============================

    init_var = tf.global_variables_initializer()
    print('[train] training_epochs = %s' % train_config.training_epochs)
    print('------------------------------------')



    # build dataset ========================

    # inputs_test_op, true_heatmap_test_op =  dataset_test_iterator.get_next()
    inputs_train_op, true_heatmap_train_op =  dataset_train_iterator.get_next()

    with tf.Session() as sess:
        # Run the variable initializer
        sess.run(init_var)

        # train_handle    = sess.run(dataset_train_iterator.string_handle())
        # test_handle     = sess.run(dataset_test_iterator.string_handle())

        for epoch in range(train_config.training_epochs):

            inputs_train,true_heatmap_train = sess.run([inputs_train_op,true_heatmap_train_op])
            # inputs_valid,true_heatmap_valid  = sess.run([inputs_test_op,true_heatmap_test_op])

            train_start_time = time.time()

            # train model
            # _,loss_train = sess.run([train_op,loss_op],
            #                          feed_dict={dataset_handle: train_handle,
            #                          modelbuilder.dropout_keeprate:model_config.output.dropout_keeprate})

            _,loss_train = sess.run([train_op,loss_op],
                                     feed_dict={inputs: inputs_train,
                                                true_heatmap: true_heatmap_train,
                                                modelbuilder.dropout_keeprate:model_config.output.dropout_keeprate})

            train_elapsed_time = time.time() - train_start_time

            global_step_eval = global_step.eval()

            if train_config.display_step == 0:
                continue
            elif global_step_eval % train_config.display_step == 0:
                print('[train] curr epochs = %s' % epoch)

                # # test model
                # loss_test = loss_op.eval(feed_dict={dataset_handle: test_handle,
                #                                     modelbuilder.dropout_keeprate: 1.0})
                #
                # loss_test = loss_op.eval( feed_dict={inputs: inputs_valid,
                #                                     true_heatmap: true_heatmap_valid,
                #                                     modelbuilder.dropout_keeprate: 1.0})

                # tf summary
                summary_loss_train = tb_summary_loss_train.eval(feed_dict={inputs: inputs_train,
                                                                            true_heatmap: true_heatmap_train,
                                                                            modelbuilder.dropout_keeprate: 1.0})
                # summary_loss_test  = tb_summary_loss_test.eval( feed_dict={inputs: inputs_valid,
                #                                                             true_heatmap: true_heatmap_valid,
                #                                                             modelbuilder.dropout_keeprate: 1.0})
                #

                # summary_loss_train = tb_summary_loss_train.eval(feed_dict={dataset_handle: train_handle,
                #                                                            modelbuilder.dropout_keeprate:1.0})
                #
                # summary_loss_test  = tb_summary_loss_test.eval(feed_dict={dataset_handle: test_handle,
                #                                                           modelbuilder.dropout_keeprate: 1.0})

                summary_lr         = tb_summary_lr.eval()

                file_writer.add_summary(summary_loss_train,global_step_eval)
                # file_writer.add_summary(summary_loss_test,global_step_eval)
                file_writer.add_summary(summary_lr,global_step_eval)

                print('At step = %d, train elapsed_time = %.1f ms' % (global_step_eval, train_elapsed_time))
                print("Training set loss (avg over batch)= %.2f   " % (loss_train))
                # print("Test set Err loss (total batch)= %.2f %%" % (loss_test))
                print("--------------------------------------------")


        print("Training finished!")

    file_writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)


    # dataloader instance gen
    dataloader_train, dataloader_test = \
        [DataLoader(
        is_training     =is_training,
        data_dir        =DATASET_DIR,
        transpose_input =False,
        use_bfloat16    =False) for is_training in [True, False]]



    dataset_train = dataloader_train.input_fn()
    # dataset_test  = dataloader_test.input_fn()
    dataset_test=None

    # model tranining
    with tf.name_scope(name='trainer'):
        # < complete the train() function call >
        train(dataset_train=dataset_train,
              dataset_test=dataset_test)

