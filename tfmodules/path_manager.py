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
# ! /usr/bin/env python
'''
    filename: path_manager.py
    description: this module include all path information on this proj

    - Author : jaewook Kang @ 20180613

'''

from os import getcwd
from os import chdir

# move to project home directory
chdir('..')

PROJ_HOME               = getcwd()
TF_MODULE_DIR           = PROJ_HOME              + '/tfmodules'

print("[pathmanager] PROJ HOME = %s" % PROJ_HOME)
# tf module related directory
EXPORT_DIR              = TF_MODULE_DIR          + '/export'
COCO_DATALOAD_DIR       = TF_MODULE_DIR          + '/coco_dataload_modules'


# data path
DATASET_DIR                 = PROJ_HOME     + '/dataset/ai_challenger'
COCO_TRAINSET_DIR            = DATASET_DIR     + '/train/'
COCO_VALIDSET_DIR            = DATASET_DIR     + '/valid/'

print("[pathmanager] DATASET_DIR = %s" % DATASET_DIR)
print("[pathmanager] COCO_DATALOAD_DIR = %s" % COCO_DATALOAD_DIR)
print("[pathmanager] EXPORT_DIR = %s" % EXPORT_DIR)


