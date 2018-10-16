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
import time
import sys

from path_manager import DATASET_DIR
from path_manager import EXPORT_DIR
from path_manager import TF_MODULE_DIR
from path_manager import COCO_DATALOAD_DIR


sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,EXPORT_DIR)
sys.path.insert(0,COCO_DATALOAD_DIR)


from model_builder import ModelBuilder
from utils         import metric_fn
from utils         import summary_eval_fn



